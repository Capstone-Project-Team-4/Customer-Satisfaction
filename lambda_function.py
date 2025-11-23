import json
import boto3
import xml.etree.ElementTree as ET
import csv
import io
import datetime
import os
import time
 
# existing clients
s3 = boto3.client("s3")
glue = boto3.client("glue")
cw = boto3.client("cloudwatch")
 
# Bedrock client (used only when ENABLE_BEDROCK == "true")
ENABLE_BEDROCK = os.getenv("ENABLE_BEDROCK", "false").lower() == "true"
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "")              # e.g. "amazon.titan-text-xx" or leave empty
# You may optionally use different models per task:
BEDROCK_MODEL_SENTIMENT = os.getenv("BEDROCK_MODEL_SENTIMENT", BEDROCK_MODEL_ID)
BEDROCK_MODEL_CHURN = os.getenv("BEDROCK_MODEL_CHURN", BEDROCK_MODEL_ID)
BEDROCK_MODEL_RECOMMEND = os.getenv("BEDROCK_MODEL_RECOMMEND", BEDROCK_MODEL_ID)
 
if ENABLE_BEDROCK:
    # Use "bedrock-runtime" client
    bedrock = boto3.client("bedrock-runtime")
 
 
GLUE_JOB_NAME = "Customer-360"
 
 
# ---------------------------
# CloudWatch helpers (extended to include bedrock metrics)
# ---------------------------
def put_cw_metric(namespace, name, value, unit="Count"):
    try:
        cw.put_metric_data(
            Namespace=namespace,
            MetricData=[{"MetricName": name, "Value": value, "Unit": unit}]
        )
    except Exception as e:
        print("Failed to publish CW metric:", e)
 
 
def report_validation_failure():
    put_cw_metric("Customer360", "LambdaValidationFailures", 1)
 
 
def report_bedrock_invocation(success=True):
    put_cw_metric("Customer360/Bedrock", "BedrockInvocations", 1)
    if not success:
        put_cw_metric("Customer360/Bedrock", "BedrockInvokeFailures", 1)
 
 
# ---------------------------
# Bedrock invocation helper
# ---------------------------
def invoke_bedrock(model_id, prompt, timeout_seconds=30):
    """
    Calls Bedrock invoke_model. Expects the model to return JSON-like text (recommended).
    Returns: (success_bool, text_response)
    """
    if not ENABLE_BEDROCK:
        print("Bedrock disabled by env var; skipping call.")
        return False, "bedrock-disabled"
 
    if not model_id:
        print("No model id configured; skipping bedrock call.")
        return False, "no-model-configured"
 
    try:
        # Some Bedrock models accept raw text body. We supply a text prompt.
        # For more advanced usage you can pass JSON; this depends on model.
        response = bedrock.invoke_model(
            modelId=model_id,
            body=prompt.encode("utf-8")  # bytes
        )
        # response['body'] is a streaming body object in many SDK versions
        body = b""
        if "body" in response:
            stream = response["body"]
            try:
                # stream might be a botocore response streaming object
                body = stream.read()
            except Exception:
                # fallback if it's already bytes/string
                body = stream
 
        if isinstance(body, bytes):
            text = body.decode("utf-8", errors="ignore")
        else:
            text = str(body)
 
        report_bedrock_invocation(success=True)
        return True, text
 
    except Exception as e:
        print("Bedrock invocation failed:", e)
        report_bedrock_invocation(success=False)
        return False, str(e)
 
 
# ---------------------------
# Task-specific prompt wrappers (simple, return parsed JSON or raw)
# ---------------------------
def predict_sentiment_from_feedback(feedback_list):
    """
    feedback_list: list of feedback strings.
    Returns: list of dicts {feedback:..., sentiment: 'positive'|'negative'|'neutral', score: 0.0}
    """
    # Create concise prompt and request JSON output
    prompt = {
        "task": "sentiment_analysis",
        "instructions": (
            "For each feedback entry return a JSON array where each element contains "
            "fields: feedback, sentiment (positive/neutral/negative), score (0-1). "
            "Return only JSON."
        ),
        "examples_count": min(len(feedback_list), 10),
        "inputs": feedback_list
    }
    # Flatten prompt to string; many Bedrock models expect text prompt
    prompt_text = json.dumps(prompt, ensure_ascii=False)
    ok, text = invoke_bedrock(BEDROCK_MODEL_SENTIMENT, prompt_text)
    if not ok:
        return [{"feedback": f, "sentiment": "unknown", "score": 0.0, "error": text} for f in feedback_list]
 
    # try parse JSON from model output
    try:
        parsed = json.loads(text)
        # basic validation
        if isinstance(parsed, list):
            return parsed
        else:
            return [{"feedback": f, "sentiment": "unknown", "score": 0.0, "raw": text} for f in feedback_list]
    except Exception:
        # best-effort: return raw output attached to each
        return [{"feedback": f, "sentiment": "unknown", "score": 0.0, "raw": text} for f in feedback_list]
 
 
def predict_churn_for_customers(customer_purchase_histories):
    """
    customer_purchase_histories: list of dicts {customer_id:..., history: ...}
    Returns: list of dicts {customer_id, churn_probability, reason}
    """
    prompt = {
        "task": "churn_prediction",
        "instructions": (
            "Given each customer's purchase history, return a JSON array with fields: "
            "customer_id, churn_probability (0-1), top_reason (short). Return only JSON."
        ),
        "inputs": customer_purchase_histories
    }
    prompt_text = json.dumps(prompt, ensure_ascii=False)
    ok, text = invoke_bedrock(BEDROCK_MODEL_CHURN, prompt_text)
    if not ok:
        return [{"customer_id": p.get("customer_id"), "churn_probability": None, "error": text} for p in customer_purchase_histories]
 
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [{"raw": text}]
    except Exception:
        return [{"raw": text}]
 
 
def recommend_products_for_profile(customer_profiles, top_k=5):
    """
    customer_profiles: list of dicts {customer_id, profile_info}
    Returns: list of dicts {customer_id, recommendations: [productIds or descriptions]}
    """
    prompt = {
        "task": "recommendation",
        "instructions": f"For each profile return top {top_k} product recommendations as JSON with fields: customer_id, recommendations (array).",
        "inputs": customer_profiles
    }
    prompt_text = json.dumps(prompt, ensure_ascii=False)
    ok, text = invoke_bedrock(BEDROCK_MODEL_RECOMMEND, prompt_text)
    if not ok:
        return [{"customer_id": p.get("customer_id"), "recommendations": [], "error": text} for p in customer_profiles]
 
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [{"raw": text}]
    except Exception:
        return [{"raw": text}]
 
 
# ---------------------------
# Your existing validators (unchanged)
# ---------------------------
def is_valid_date(date_str):
    try:
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except:
        return False
 
 
def process_xml_records(content):
    root = ET.fromstring(content)
    valid_customers = []
    invalid_customers = []
    for cust in root.findall(".//Customer"):
        cid = cust.find("CustomerID")
        name = cust.find("Name")
        city = cust.find("City")
        if cid is None or name is None or city is None:
            invalid_customers.append(cust)
            continue
        if not cid.text or not cid.text.strip().isdigit():
            invalid_customers.append(cust)
            continue
        if not name.text or name.text.strip() == "":
            invalid_customers.append(cust)
            continue
        if not city.text or city.text.strip() == "":
            invalid_customers.append(cust)
            continue
        valid_customers.append(cust)
    return valid_customers, invalid_customers
 
 
def build_xml(customers):
    root = ET.Element("Customers")
    for cust in customers:
        root.append(cust)
    return ET.tostring(root, encoding="utf-8")
 
 
def process_json_records(content):
    records = json.loads(content)
    valid = []
    invalid = []
    for rec in records:
        try:
            if ("CustomerID" not in rec or not str(rec["CustomerID"]).isdigit() or
                "Amount" not in rec or float(rec["Amount"]) <= 0 or
                "Product" not in rec or rec["Product"].strip() == "" or
                "Date" not in rec or not is_valid_date(rec["Date"])):
                invalid.append(rec)
            else:
                valid.append(rec)
        except:
            invalid.append(rec)
    return valid, invalid
 
 
def process_csv_records(content):
    f = io.StringIO(content)
    reader = csv.DictReader(f)
    valid = []
    invalid = []
    for row in reader:
        try:
            if (not row["CustomerID"].isdigit() or
                int(row["Rating"]) < 1 or int(row["Rating"]) > 5 or
                row["Feedback"].strip() == ""):
                invalid.append(row)
            else:
                valid.append(row)
        except:
            invalid.append(row)
    return valid, invalid
 
 
# ---------------------------
# MAIN HANDLER (original flow with added Bedrock plugs)
# ---------------------------
def lambda_handler(event, context):
    # existing event handling (will still error if event lacks Records)
    try:
        record = event["Records"][0]
    except Exception as e:
        print("ERROR: 'Records' missing or invalid event:", e)
        report_validation_failure()
        return {"status": "No Records"}
 
    bucket = record["s3"]["bucket"]["name"]
    key = record["s3"]["object"]["key"]
    print(f"Processing: {key}")
 
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read().decode("utf-8")
    except Exception as e:
        print("Failed to read object:", e)
        report_validation_failure()
        return {"status": "Failed to read file"}
 
    total_valid = 0
    total_invalid = 0
 
    # we will optionally collect data for bedrock
    feedback_texts_for_bedrock = []
    customer_profiles_for_recommend = []
    customer_histories_for_churn = []
 
    # --------------------- Process current file ---------------------
    if key.endswith(".xml"):
        valid, invalid = process_xml_records(content)
        if valid:
            validated_xml = build_xml(valid)
            s3.put_object(Bucket=bucket, Key=key.replace("raw/", "validated/"), Body=validated_xml)
            total_valid += len(valid)
        if invalid:
            error_xml = build_xml(invalid)
            s3.put_object(Bucket=bucket, Key=key.replace("raw/", "error/"), Body=error_xml)
            total_invalid += len(invalid)
 
        # Example: if XML contains customer profiles, prepare for recommendations
        for cust in valid:
            try:
                cid = cust.find("CustomerID").text
                # collect profile snippet - customize as needed
                profile_snippet = {
                    "customer_id": cid,
                    "profile_info": {
                        "name": cust.find("Name").text if cust.find("Name") is not None else "",
                        "city": cust.find("City").text if cust.find("City") is not None else ""
                    }
                }
                customer_profiles_for_recommend.append(profile_snippet)
            except:
                pass
 
    elif key.endswith(".json"):
        valid, invalid = process_json_records(content)
        if valid:
            s3.put_object(Bucket=bucket, Key=key.replace("raw/", "validated/"), Body=json.dumps(valid, indent=2))
            total_valid += len(valid)
        if invalid:
            s3.put_object(Bucket=bucket, Key=key.replace("raw/", "error/"), Body=json.dumps(invalid, indent=2))
            total_invalid += len(invalid)
 
        # For JSON purchase histories, prepare churn input
        for rec in valid:
            try:
                # example expected structure; adjust to your actual schema
                customer_histories_for_churn.append({
                    "customer_id": rec.get("CustomerID"),
                    "history": {
                        "purchases": rec.get("Purchases", []),
                        "total_spend": rec.get("Amount", 0)
                    }
                })
            except:
                pass
 
    elif key.endswith(".csv"):
        valid, invalid = process_csv_records(content)
        if valid:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=valid[0].keys())
            writer.writeheader()
            writer.writerows(valid)
            s3.put_object(Bucket=bucket, Key=key.replace("raw/", "validated/"), Body=output.getvalue())
            total_valid += len(valid)
        if invalid:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=invalid[0].keys())
            writer.writeheader()
            writer.writerows(invalid)
            s3.put_object(Bucket=bucket, Key=key.replace("raw/", "error/"), Body=output.getvalue())
            total_invalid += len(invalid)
 
        # For feedback CSV, collect Feedback text for sentiment
        for row in valid:
            try:
                if "Feedback" in row and row["Feedback"].strip() != "":
                    feedback_texts_for_bedrock.append(row["Feedback"].strip())
            except:
                pass
 
    # ------------------------- Fetch previously validated & error objects (unchanged) -------------------------
    validated_objects = s3.list_objects_v2(Bucket=bucket, Prefix="validated/")
    for obj in validated_objects.get('Contents', []):
        file_key = obj["Key"]
        obj2 = s3.get_object(Bucket=bucket, Key=file_key)
        content2 = obj2["Body"].read().decode("utf-8")
 
        if file_key.endswith(".xml"):
            valid2, _ = process_xml_records(content2)
        elif file_key.endswith(".json"):
            valid2, _ = process_json_records(content2)
        elif file_key.endswith(".csv"):
            valid2, _ = process_csv_records(content2)
        else:
            continue
 
        total_valid += len(valid2)
 
    error_objects = s3.list_objects_v2(Bucket=bucket, Prefix="error/")
    for obj in error_objects.get('Contents', []):
        file_key = obj["Key"]
        obj2 = s3.get_object(Bucket=bucket, Key=file_key)
        content2 = obj2["Body"].read().decode("utf-8")
 
        if file_key.endswith(".xml"):
            _, invalid2 = process_xml_records(content2)
        elif file_key.endswith(".json"):
            _, invalid2 = process_json_records(content2)
        elif file_key.endswith(".csv"):
            _, invalid2 = process_csv_records(content2)
        else:
            continue
 
        total_invalid += len(invalid2)
 
    # ------------------------- Compute Data Quality Score -------------------------
    dq_score = (
        (total_valid / (total_valid + total_invalid)) * 100
        if (total_valid + total_invalid) > 0
        else 0
    )
 
    if total_invalid > 0:
        report_validation_failure()
 
    # ------------------------- Bedrock: run predictive tasks (batch) -------------------------
    # Only run if bedrock is enabled and we have relevant items. Keep this batched to control costs.
    try:
        if ENABLE_BEDROCK:
            # 1) Sentiment analysis for feedbacks
            if feedback_texts_for_bedrock:
                print("Calling Bedrock for sentiment (batched)...")
                sentiments = predict_sentiment_from_feedback(feedback_texts_for_bedrock)
                # store results back to S3 for later consumption
                s3.put_object(Bucket=bucket, Key=key.replace("raw/", "ai/sentiment/") + ".json", Body=json.dumps(sentiments, indent=2))
 
            # 2) Churn prediction
            if customer_histories_for_churn:
                print("Calling Bedrock for churn predictions (batched)...")
                churn_preds = predict_churn_for_customers(customer_histories_for_churn)
                s3.put_object(Bucket=bucket, Key=key.replace("raw/", "ai/churn/") + ".json", Body=json.dumps(churn_preds, indent=2))
 
            # 3) Recommendations
            if customer_profiles_for_recommend:
                print("Calling Bedrock for recommendations (batched)...")
                recs = recommend_products_for_profile(customer_profiles_for_recommend, top_k=5)
                s3.put_object(Bucket=bucket, Key=key.replace("raw/", "ai/recommendations/") + ".json", Body=json.dumps(recs, indent=2))
    except Exception as e:
        print("Error during Bedrock predictive steps:", e)
        # Bedrock invocation counters already handle metrics; do not bubble up as Lambda error.
 
    # ------------------------- Trigger GLUE JOB (if valid records exist) -------------------------
    if total_valid > 0:
        glue.start_job_run(
            JobName=GLUE_JOB_NAME,
            Arguments={
                "--JOB_NAME": "Customer-360",
                "--VALIDATED_XML_PATH": "s3://customer-360-datalake/validated/customer_details/",
                "--VALIDATED_JSON_PATH": "s3://customer-360-datalake/validated/customer_purchases/",
                "--VALIDATED_CSV_PATH": "s3://customer-360-datalake/validated/customer_feedback/",
                "--DYNAMO_TABLE": "customer360_etl_tracking",
                "--RDS_JDBC_URL": "jdbc:postgresql://customer360-db.cw3aw0iuuvtz.us-east-1.rds.amazonaws.com:5432/postgres",
                "--RDS_USER": "postgres",
                "--RDS_PASSWORD": "customer360!",
                "--DQ_SCORE": str(dq_score),
                "--ERROR_COUNT": str(total_invalid)
            }
        )
 
    return {"status": "Record-level processing complete"}
