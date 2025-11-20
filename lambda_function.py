import json
import boto3
import xml.etree.ElementTree as ET
import csv
import io
import datetime

s3 = boto3.client("s3")
glue = boto3.client("glue")
cw = boto3.client("cloudwatch")

GLUE_JOB_NAME = "Customer-360"

# ---------------------------------------------------------
# SEND CUSTOM METRICS TO CLOUDWATCH
# ---------------------------------------------------------

def report_validation_failure():
    cw.put_metric_data(
        Namespace="Customer360",
        MetricData=[
            {
                "MetricName": "LambdaValidationFailures",
                "Value": 1
            }
        ]
    )


# --------------------------
# Helpers
# --------------------------

def is_valid_date(date_str):
    try:
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except:
        return False


# ---------------------------------------------------------
# XML RECORD-LEVEL VALIDATION
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# JSON RECORD-LEVEL VALIDATION
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# CSV RECORD-LEVEL VALIDATION
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# MAIN HANDLER
# ---------------------------------------------------------

def lambda_handler(event, context):

    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    key = record["s3"]["object"]["key"]

    print(f"Processing: {key}")

    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read().decode("utf-8")
    except:
        report_validation_failure()
        return {"status": "Failed to read file"}

    total_valid = 0
    total_invalid = 0

    # ---------------------
    # Process current file
    # ---------------------
    if key.endswith(".xml"):
        valid, invalid = process_xml_records(content)

        if valid:
            validated_xml = build_xml(valid)
            s3.put_object(
                Bucket=bucket,
                Key=key.replace("raw/", "validated/"),
                Body=validated_xml
            )
            total_valid += len(valid)

        if invalid:
            error_xml = build_xml(invalid)
            s3.put_object(
                Bucket=bucket,
                Key=key.replace("raw/", "error/"),
                Body=error_xml
            )
            total_invalid += len(invalid)

    elif key.endswith(".json"):
        valid, invalid = process_json_records(content)

        if valid:
            s3.put_object(
                Bucket=bucket,
                Key=key.replace("raw/", "validated/"),
                Body=json.dumps(valid, indent=2)
            )
            total_valid += len(valid)

        if invalid:
            s3.put_object(
                Bucket=bucket,
                Key=key.replace("raw/", "error/"),
                Body=json.dumps(invalid, indent=2)
            )
            total_invalid += len(invalid)

    elif key.endswith(".csv"):
        valid, invalid = process_csv_records(content)

        if valid:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=valid[0].keys())
            writer.writeheader()
            writer.writerows(valid)
            s3.put_object(
                Bucket=bucket,
                Key=key.replace("raw/", "validated/"),
                Body=output.getvalue()
            )
            total_valid += len(valid)

        if invalid:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=invalid[0].keys())
            writer.writeheader()
            writer.writerows(invalid)
            s3.put_object(
                Bucket=bucket,
                Key=key.replace("raw/", "error/"),
                Body=output.getvalue()
            )
            total_invalid += len(invalid)

    # -------------------------
    # Fetch All Valid + Invalid
    # -------------------------
    validated_objects = s3.list_objects_v2(Bucket=bucket, Prefix="validated/")
    for obj in validated_objects.get('Contents', []):
        file_key = obj["Key"]
        obj = s3.get_object(Bucket=bucket, Key=file_key)
        content = obj["Body"].read().decode("utf-8")

        if file_key.endswith(".xml"):
            valid, _ = process_xml_records(content)
        elif file_key.endswith(".json"):
            valid, _ = process_json_records(content)
        elif file_key.endswith(".csv"):
            valid, _ = process_csv_records(content)
        else:
            continue

        total_valid += len(valid)

    error_objects = s3.list_objects_v2(Bucket=bucket, Prefix="error/")
    for obj in error_objects.get('Contents', []):
        file_key = obj["Key"]
        obj = s3.get_object(Bucket=bucket, Key=file_key)
        content = obj["Body"].read().decode("utf-8")

        if file_key.endswith(".xml"):
            _, invalid = process_xml_records(content)
        elif file_key.endswith(".json"):
            _, invalid = process_json_records(content)
        elif file_key.endswith(".csv"):
            _, invalid = process_csv_records(content)
        else:
            continue

        total_invalid += len(invalid)

    # -------------------------
    # Compute Data Quality Score
    # -------------------------
    dq_score = (
        (total_valid / (total_valid + total_invalid)) * 100
        if (total_valid + total_invalid) > 0
        else 0
    )

    # -------------------------
    # REPORT FAILURES TO CLOUDWATCH
    # -------------------------
    if total_invalid > 0:
        report_validation_failure()

    # -------------------------
    # TRIGGER GLUE JOB (if valid records exist)
    # -------------------------
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
