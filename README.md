Here's a Markdown formatted README file for GitHub:

```markdown
# Customer 360 Data Pipeline on AWS

This project builds a complete Customer 360 data pipeline on AWS that processes customer information from three file formats: XML for customer profiles, JSON for purchases, and CSV for feedback. The pipeline validates incoming files, separates good files from bad files, performs scalable PySpark transformations in AWS Glue, tracks ETL runs in DynamoDB, loads the final data into a PostgreSQL database on Amazon RDS, enhances feedback with sentiment labels, and provides reporting through Amazon QuickSight. The system is monitored by CloudWatch and alerts via EventBridge and SNS on failures.

## Prerequisites

Before starting, ensure you have:
- An AWS account
- Basic understanding of AWS services: S3, IAM, Lambda, Glue, DynamoDB, RDS
- Data files:
  - **XML**: Customer details
  - **JSON**: Purchases
  - **CSV**: Feedback
- PostgreSQL client like pgAdmin or DBeaver

## Steps

### Step 1: Create S3 Bucket

**Bucket Creation**
1. Open AWS Console → S3 → Create bucket
2. Name: `customer-360-datalake`
3. Region: Same as all your other services

**Folder Structure**
Inside the bucket, create the following folders:
```
raw/
    customer_details/
    purchases/
    feedback/
validated/
    customer_details/
    purchases/
    feedback/
error/
    customer_details/
    purchases/
    feedback/
```

**Upload Files**
- Upload your XML to `raw/customer_details`
- Upload JSON to `raw/purchases`
- Upload CSV to `raw/feedback`

### Step 2: IAM Policies and Roles

**Create the following IAM roles:**

**a)** *Lambda Execution Role*  
Must include permissions for:
- S3 read/write
- DynamoDB write
- Glue: StartJobRun
- CloudWatch Logs

**b)** *Glue Job Role*  
Must include:
- AmazonS3FullAccess 
- AmazonRDSFullAccess
- CloudWatch Logs

### Step 3: Create AWS Lambda Function

**Lambda is responsible for:**
- Receiving S3 event
- Identifying whether file is XML/JSON/CSV
- Validating schema
- Moving valid data to `validated/`
- Moving invalid data to `error/`
- Triggering Glue ETL Job

**Steps to create Lambda:**
1. Go to AWS Console → Lambda
2. Create function → Author from scratch
3. Choose Python 3.12
4. Attach the IAM role created earlier
5. Add an S3 trigger
6. Upload your Lambda code
7. Deploy

**Connect S3 to Lambda**
1. Go to S3 bucket → Properties → Event Notifications
2. Create notification:
   - Prefix: `raw/`
   - Event type: All object create events
   - Destination: Lambda function

### Step 4: Create AWS Glue Job (PySpark ETL)

**Glue performs:**
- Reading validated XML, JSON, CSV
- Transformations using PySpark
- Joins & aggregations
- Data Quality (DQ) score computation
- Writing final curated data to RDS

**Steps:**
1. Go to AWS Console → AWS Glue
2. Create job → “Spark” job
3. Choose Glue version 
4. Assign Glue Role
5. Paste the PySpark code
6. Set S3 temporary directory
7. Save & run

### Step 5: Create DynamoDB Table

**DynamoDB is used to store the final Data Quality (DQ) results produced by the Glue job.**

**DynamoDB is responsible for:**
- Storing each file’s DQ score
- Storing error count
- Storing validation status (SUCCESS / FAILED)
- Storing paths of validated XML / JSON / CSV files
- Storing timestamp of when Glue processed the file
- Providing a historical log for reporting, auditing, and monitoring

**Steps to Create DynamoDB Table:**
1. Go to AWS Console → DynamoDB
2. Click Create table
3. Enter table name (example: `Customer360_table`)
4. Set Partition Key: `record_id` (String)
5. Keep default settings (no need for sort key or indexes)
6. Create table

### Step 6: Set Up Amazon RDS (PostgreSQL)

**Steps:**
1. Go to AWS Console → RDS
2. Create database → PostgreSQL
3. DB Name: `customer360`
4. Instance: `db.t3.micro` (low cost)
5. Create a security group allowing connections from your IP

**Connect RDS to pgAdmin/DBeaver**
- Use:
  - Host
  - Port: 5432
  - Username
  - Password

### Step 7: Build Dashboards Using QuickSight

**Steps:**
1. Go to AWS Console → QuickSight
2. Create new dataset
3. Choose “PostgreSQL”
4. Connect to your RDS database
5. Import `customer360_golden`
6. Build dashboards such as:
   - Total Spend
   - Avg Rating
   - Sentiment distribution
   - Customer Profiles

### Step 8: Monitoring & Alerts

**A. CloudWatch Alarms**
Create alarms for:
- Lambda errors
- RDS write failures
- High DQ failure percentage

**Steps:**
1. Go to CloudWatch → Alarms → Create Alarm
2. Choose metric: Lambda → Errors
3. Set threshold (e.g., > 5)
4. Send notification to SNS

**B. EventBridge for Glue Job Failure**
1. Go to EventBridge → Create rule
2. Pattern:
   - Event source: “Glue”
   - Detail type: “Job Run State Change”
3. Target: SNS topic

**C. SNS Notifications**
1. Go to SNS → Create Topic → Standard
2. Add your email subscription
3. Confirm email
4. Connect SNS with CloudWatch + EventBridge

## Clean up Resources

You can delete the resources created for this project unless you want to retain them. This prevents unnecessary charges to your AWS account.

**Delete the following AWS resources:**
- Lambda Function
- Glue Job
- DynamoDB Table 
- RDS PostgreSQL Instance
- S3 Buckets:
  - Raw bucket
  - Validated bucket
  - Output/processed bucket
  - Any log/temporary buckets you created
- EventBridge Rule for Glue Job Failure Alerts
- CloudWatch Alarms:
  - Lambda error alarm
  - Glue job failure alarm
  - RDS alarm
  - DQ failure percentage alarm
  - Any custom alarms you created
- SNS Topic used for notifications
```

This README provides a clear and structured guide for setting up and working with the Customer 360 data pipeline on AWS.
