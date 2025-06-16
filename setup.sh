#!/bin/bash

# === CONFIGURATION ===
PROFILE_NAME="new-account"  # Change if you want a different profile name

# === PROMPT FOR CREDENTIALS ===
read -p "Enter AWS Access Key ID: " AWS_ACCESS_KEY_ID
read -s -p "Enter AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
echo ""
read -p "Enter default region (e.g., us-east-1): " AWS_REGION
read -p "Enter output format [json|table|text] (default: json): " AWS_OUTPUT
AWS_OUTPUT=${AWS_OUTPUT:-json}

# === CONFIGURE PROFILE ===
aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID" --profile "$PROFILE_NAME"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY" --profile "$PROFILE_NAME"
aws configure set region "$AWS_REGION" --profile "$PROFILE_NAME"
aws configure set output "$AWS_OUTPUT" --profile "$PROFILE_NAME"

# === VERIFY ===
echo ""
echo "Verifying the configured AWS account..."
aws sts get-caller-identity --profile "$PROFILE_NAME"

# === OPTIONAL: EXPORT FOR CURRENT SESSION ===
read -p "Do you want to use this profile as default for this terminal session? (y/n): " CONFIRM
if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
  export AWS_PROFILE="$PROFILE_NAME"
  echo "AWS_PROFILE is now set to '$PROFILE_NAME' for this session."
  echo "To make it permanent, add this to your ~/.bashrc or ~/.zshrc:"
  echo "export AWS_PROFILE=$PROFILE_NAME"
else
  echo "You can run commands like: aws s3 ls --profile $PROFILE_NAME"
fi
