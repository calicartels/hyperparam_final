import fs from 'fs';
import path from 'path';

/**
 * Set up Google Cloud credentials
 */
export function setupGoogleCloudCredentials(): void {
  // If we have the service account key in environment variables, use it directly
  if (process.env.GOOGLE_SERVICE_ACCOUNT_KEY) {
    console.log('Using Google service account key from environment variable');
    
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
      fs.writeFileSync(serviceAccountPath, process.env.GOOGLE_SERVICE_ACCOUNT_KEY);
      console.log(`Wrote service account key to ${serviceAccountPath}`);
      
      // Set the environment variable for Google Cloud authentication
      process.env.GOOGLE_APPLICATION_CREDENTIALS = serviceAccountPath;
      console.log(`Set GOOGLE_APPLICATION_CREDENTIALS to ${serviceAccountPath}`);
    } catch (error) {
      console.error('Error writing service account key file:', error);
    }
  } else {
    console.log('No Google service account key found in environment variables');
  }
}