/**
 * ReturnWise AI - Amazon S3 Storage Adapter
 * =========================================
 * Provides optional synchronization for machine learning model artifacts
 * and datasets with Amazon S3.
 * Defaults to local filesystem storage if S3 credentials are not configured.
 */

const fs = require("fs");
const path = require("path");
const { S3Client, PutObjectCommand, GetObjectCommand } = require("@aws-sdk/client-s3");

class S3Service {
  constructor() {
    this.bucketName = process.env.S3_BUCKET_NAME || "";
    this.region = process.env.S3_REGION || process.env.AWS_REGION || "us-east-1";
    this.accessKeyId = process.env.AWS_ACCESS_KEY_ID || "";
    this.secretAccessKey = process.env.AWS_SECRET_ACCESS_KEY || "";
    this.sessionToken = process.env.AWS_SESSION_TOKEN || "";

    this.isConfigured = Boolean(
      this.bucketName &&
      this.accessKeyId &&
      this.secretAccessKey &&
      !this.accessKeyId.startsWith("your_")
    );

    this.client = null;
    if (this.isConfigured) {
      try {
        const clientConfig = {
          region: this.region,
          credentials: {
            accessKeyId: this.accessKeyId,
            secretAccessKey: this.secretAccessKey,
            ...(this.sessionToken ? { sessionToken: this.sessionToken } : {})
          }
        };
        this.client = new S3Client(clientConfig);
      } catch (err) {
        console.warn("[S3] Failed to initialize S3 client:", err.message);
        this.client = null;
      }
    }
  }

  getStatus() {
    return {
      configured: this.isConfigured,
      bucket: this.bucketName || "Not configured (using local files)",
      region: this.region,
      storageMode: this.isConfigured ? "S3-Enabled" : "Local Filesystem"
    };
  }

  /**
   * Uploads a local file to S3 if configured.
   */
  async uploadFile(localFilePath, s3Key) {
    if (!this.isConfigured || !this.client) {
      return {
        success: false,
        message: "S3 upload skipped: S3_BUCKET_NAME or credentials not configured. Using local file.",
        localPath: localFilePath
      };
    }

    try {
      const fileContent = fs.readFileSync(localFilePath);
      const command = new PutObjectCommand({
        Bucket: this.bucketName,
        Key: s3Key,
        Body: fileContent
      });

      await this.client.send(command);
      return {
        success: true,
        message: `Successfully uploaded ${path.basename(localFilePath)} to s3://${this.bucketName}/${s3Key}`,
        s3Uri: `s3://${this.bucketName}/${s3Key}`
      };
    } catch (err) {
      console.warn(`[S3] Upload error for ${s3Key}:`, err.message);
      return {
        success: false,
        message: `S3 upload failed: ${err.message}`,
        error: err.message
      };
    }
  }
}

module.exports = new S3Service();
