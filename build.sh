#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Please provide the solution name as well as the base S3 bucket name and the region to run build script."
    echo "For example: ./build.sh trademarked-solution-name sagemaker-solutions-build us-west-2"
    exit 1
fi

# Package the solution assistant
mkdir build
mkdir build/solution-assistant
cp -r ./cloudformation/solution-assistant ./build/
(cd ./build/solution-assistant && pip install -r requirements.txt -t ./src/site-packages)
find ./build/solution-assistant -name '*.pyc' -delete
(cd ./build/solution-assistant/src && zip -q -r9 ../../solution-assistant.zip *)
rm -rf ./build/solution-assistant

# Upload to S3
s3_prefix="s3://$2-$3/$1"
echo "Using S3 path: $s3_prefix"
aws s3 cp --recursive sagemaker $s3_prefix/sagemaker --exclude '.*' --exclude "*~"
aws s3 cp --recursive cloudformation $s3_prefix/cloudformation --exclude '.*' --exclude "*~"
aws s3 cp --recursive build $s3_prefix/build
aws s3 cp Readme.md $s3_prefix/
