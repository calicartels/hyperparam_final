import fs from 'fs';
import path from 'path';

/**
 * Set up Google Cloud credentials
 */
export function setupGoogleCloudCredentials(): boolean {
  // If we have the service account key in environment variables, use it directly
  if (process.env.GOOGLE_SERVICE_ACCOUNT_KEY) {
    console.log('Using Google service account key from environment variable');
    
    // The service account key might contain newlines or be base64-encoded
    // Let's try a few different approaches to parse it
    let serviceAccountKey = process.env.GOOGLE_SERVICE_ACCOUNT_KEY;
    let isValidJson = false;
    
    try {
      // First attempt: Try to parse as-is
      JSON.parse(serviceAccountKey);
      isValidJson = true;
    } catch (error) {
      console.log('Service account key is not valid JSON as-is, trying alternative formats');
      
      // Second attempt: Try replacing escaped newlines
      try {
        const keyWithNewlines = serviceAccountKey.replace(/\\n/g, '\n');
        JSON.parse(keyWithNewlines);
        serviceAccountKey = keyWithNewlines;
        isValidJson = true;
      } catch (error) {
        // Third attempt: Try base64 decoding
        try {
          const decoded = Buffer.from(serviceAccountKey, 'base64').toString();
          JSON.parse(decoded);
          serviceAccountKey = decoded;
          isValidJson = true;
        } catch (error) {
          console.error('Failed to parse service account key in any format');
          return false;
        }
      }
    }
    
    if (!isValidJson) {
      console.error('Invalid Google service account key format (not valid JSON)');
      return false;
    }
    
    // Create credentials directory if it doesn't exist
    const credentialsDir = path.join(process.cwd(), 'credentials');
    if (!fs.existsSync(credentialsDir)) {
      fs.mkdirSync(credentialsDir, { recursive: true });
      console.log(`Created credentials directory at ${credentialsDir}`);
    }

    // Write the key to a temporary file
    const serviceAccountPath = path.join(
      credentialsDir, 
      'google-service-account-key.json'
    );
    
    // Write the key to the file
    try {
      fs.writeFileSync(serviceAccountPath, serviceAccountKey);
      console.log(`Wrote service account key to ${serviceAccountPath}`);
      
      // Set the environment variable for Google Cloud authentication
      process.env.GOOGLE_APPLICATION_CREDENTIALS = serviceAccountPath;
      console.log(`Set GOOGLE_APPLICATION_CREDENTIALS to ${serviceAccountPath}`);
      
      // Validate required Google Cloud information
      if (!process.env.GOOGLE_PROJECT_ID) {
        console.error('Missing required GOOGLE_PROJECT_ID environment variable');
        return false;
      }
      
      return true;
    } catch (error) {
      console.error('Error writing service account key file:', error);
      return false;
    }
  } else {
    console.log('No Google service account key found in environment variables');
    return false;
  }
}