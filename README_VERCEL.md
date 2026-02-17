# Deploying to Vercel

This guide explains how to deploy the Medical Bill Extractor API to Vercel.

## Prerequisites

1. **GitHub Repository**: Push your code to a GitHub repository
2. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
3. **AWS Credentials**: You'll need AWS access keys for Textract and Bedrock

## Project Structure

The application has been restructured for Vercel's serverless deployment:

```
MBR-API/
├── api/
│   └── index.py          # Vercel entry point
├── app/
│   ├── main.py           # FastAPI application
│   ├── processor.py      # Core processing logic
│   └── pricing.py        # Medicare pricing enrichment
├── vercel.json           # Vercel configuration
└── requirements.txt      # Python dependencies
```

## Deployment Steps

### 1. Push to GitHub

```bash
cd /Users/yeshwanth.penukonda/Desktop/Doclens/Antigravity_testing/MBR_API/MBR-API
git add .
git commit -m "Restructure for Vercel deployment"
git push origin main
```

### 2. Import Project to Vercel

1. Go to [vercel.com/new](https://vercel.com/new)
2. Click "Import Git Repository"
3. Select your GitHub repository
4. Vercel will auto-detect the configuration from `vercel.json`

### 3. Configure Environment Variables

In the Vercel dashboard, add these environment variables:

| Variable Name | Value | Description |
|--------------|-------|-------------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key | Required for Textract/Bedrock |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key | Required for Textract/Bedrock |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region for services |

**To add environment variables:**
1. Go to your project settings in Vercel
2. Navigate to "Environment Variables"
3. Add each variable for Production, Preview, and Development environments

### 4. Deploy

Click "Deploy" and Vercel will:
- Install Python dependencies from `requirements.txt`
- Build the serverless function
- Deploy to a production URL

## Testing the Deployment

Once deployed, test your API:

```bash
# Replace with your Vercel URL
curl https://your-project.vercel.app/api/health

# Expected response:
# {"status":"healthy"}
```

## Frontend Integration

Update your frontend to use the Vercel API URL:

```javascript
// Before (local)
const API_URL = 'http://localhost:8000';

// After (Vercel)
const API_URL = 'https://your-project.vercel.app/api';
```

## Important Notes

### Cache Behavior

- **Local Development**: Cache stored in `./cache` directory
- **Vercel Production**: Cache stored in `/tmp` (ephemeral, cleared between invocations)
- **Recommendation**: For production, consider using external storage (S3, Redis) for persistent caching

### File Size Limits

Vercel has deployment limits:
- Maximum deployment size: 250MB
- Maximum serverless function size: 50MB
- Ensure demo files in `Page_classification/` don't exceed limits

### Cold Starts

Serverless functions may experience cold starts (1-3 seconds) after periods of inactivity. The first request after deployment or inactivity will be slower.

### AWS Credentials

The application automatically detects the environment:
- **Local**: Uses `aidev` profile from `~/.aws/credentials`
- **Vercel**: Uses environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)

## Local Testing with Vercel CLI

Test the Vercel deployment locally:

```bash
# Install Vercel CLI
npm i -g vercel

# Run local development server
cd /Users/yeshwanth.penukonda/Desktop/Doclens/Antigravity_testing/MBR_API/MBR-API
vercel dev

# Test endpoints
curl http://localhost:3000/api/health
```

## Troubleshooting

### Import Errors

If you see import errors, verify the path setup in `api/index.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### AWS Authentication Errors

Verify environment variables are set correctly in Vercel dashboard. Check logs:

```bash
vercel logs your-project-url
```

### Cache Not Working

Remember that `/tmp` cache is ephemeral on Vercel. Each function invocation starts fresh. For persistent caching, implement S3 or Redis storage.

## Monitoring

View logs and analytics in the Vercel dashboard:
- **Logs**: Real-time function logs
- **Analytics**: Request counts, response times
- **Errors**: Stack traces and error rates

## Continuous Deployment

Vercel automatically deploys on every push to your main branch. To disable:
1. Go to Project Settings → Git
2. Disable "Production Branch" auto-deployment

## Support

- [Vercel Documentation](https://vercel.com/docs)
- [Vercel Python Runtime](https://vercel.com/docs/functions/serverless-functions/runtimes/python)
- [FastAPI on Vercel](https://vercel.com/guides/deploying-fastapi-with-vercel)
