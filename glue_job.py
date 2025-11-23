import sys
import boto3
import json
from datetime import datetime

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.context import SparkContext
from pyspark.sql import functions as F

# Read Glue Job Arguments

args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'VALIDATED_XML_PATH',
    'VALIDATED_JSON_PATH',
    'VALIDATED_CSV_PATH',
    'DYNAMO_TABLE',
    'RDS_JDBC_URL',
    'RDS_USER',
    'RDS_PASSWORD',
    'DQ_SCORE',
    'ERROR_COUNT'
])

job_name      = args['JOB_NAME']
xml_path      = args['VALIDATED_XML_PATH']
json_path     = args['VALIDATED_JSON_PATH']
csv_path      = args['VALIDATED_CSV_PATH']
dynamo_table  = args['DYNAMO_TABLE']
jdbc_url      = args['RDS_JDBC_URL']
rds_user      = args['RDS_USER']
rds_password  = args['RDS_PASSWORD']
dq_score      = float(args['DQ_SCORE'])
error_count   = int(args['ERROR_COUNT'])

# Spark + Glue 

sc = SparkContext()
glue_ctx = GlueContext(sc)
spark = glue_ctx.spark_session

job = Job(glue_ctx)
job.init(job_name, args)

# DynamoDB START Log

dynamodb = boto3.client('dynamodb')

job_id = f"{job_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

dynamodb.put_item(
    TableName=dynamo_table,
    Item={
        'job_id':       {'S': job_id},
        'start_time':   {'S': datetime.now().isoformat()},
        'status':       {'S': 'RUNNING'},
        'dq_score':     {'N': str(dq_score)},
        'error_count':  {'N': str(error_count)}
    }
)

# READ VALIDATED DATA

customers_df = spark.read.format("xml") \
    .option("rowTag", "Customer") \
    .load(xml_path)

purchases_df = spark.read.option("multiLine", "true").json(json_path)

feedback_df  = spark.read.csv(csv_path, header=True, inferSchema=True)

# Deduplication

customers_df = customers_df.dropDuplicates(["CustomerID"])
purchases_df = purchases_df.dropDuplicates(["CustomerID", "Product", "Date"])
feedback_df  = feedback_df.dropDuplicates(["CustomerID", "Feedback"])

valid_ids = customers_df.select("CustomerID").distinct()

purchases_df = purchases_df.join(valid_ids, "CustomerID", "inner")
feedback_df  = feedback_df.join(valid_ids, "CustomerID", "inner")

valid_records = (
    customers_df.count() +
    purchases_df.count() +
    feedback_df.count()
)

# Transformations

purchases_df = purchases_df.withColumn(
    "PurchaseDate", F.to_date("Date", "yyyy-MM-dd")
)

agg_df = purchases_df.groupBy("CustomerID").agg(
    F.sum("Amount").alias("TotalSpend"),
    F.count("*").alias("PurchaseCount"),
    F.max("PurchaseDate").alias("LastPurchaseDate")
)

avg_rating_df = feedback_df.groupBy("CustomerID").agg(
    F.avg("Rating").alias("AvgRating"),
    F.count("Feedback").alias("FeedbackCount")
)

# Build Final Customer360 Dataset

customer360_df = customers_df.alias("c") \
    .join(purchases_df.alias("p"), "CustomerID", "left") \
    .join(feedback_df.alias("f"), "CustomerID", "left") \
    .join(agg_df.alias("a"), "CustomerID", "left") \
    .join(avg_rating_df.alias("r"), "CustomerID", "left") \
    .select(
        "CustomerID",
        "Name",
        "City",
        "Product",
        "Amount",
        "Rating",
        "Feedback",
        "TotalSpend",
        "PurchaseCount",
        "AvgRating",
        "LastPurchaseDate"
    )

customer360_df = customer360_df.filter(
    F.col("CustomerID").isNotNull() &
    F.col("Amount").isNotNull() &
    F.col("TotalSpend").isNotNull() &
    F.col("PurchaseCount").isNotNull() &
    F.col("AvgRating").isNotNull() &
    F.col("LastPurchaseDate").isNotNull() &
    F.col("Name").isNotNull() &
    F.col("City").isNotNull() &
    F.col("Product").isNotNull() &
    F.col("Feedback").isNotNull() &
    F.col("Rating").isNotNull()
)

# Write to RDS

customer360_df.write \
    .format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", "customer360_golden") \
    .option("user", rds_user) \
    .option("password", rds_password) \
    .mode("overwrite") \
    .save()

# DynamoDB SUCCESS Update

dynamodb.update_item(
    TableName=dynamo_table,
    Key={'job_id': {'S': job_id}},
    UpdateExpression="""
        SET #st = :s,
            end_time = :e,
            record_count = :r,
            dq_score = :dq,
            error_count = :ec
    """,
    ExpressionAttributeNames={'#st': 'status'},
    ExpressionAttributeValues={
        ':s':  {'S': 'SUCCESS'},
        ':e':  {'S': datetime.now().isoformat()},
        ':r':  {'N': str(valid_records)},
        ':dq': {'N': str(dq_score)},
        ':ec': {'N': str(error_count)}
    }
)

job.commit()
