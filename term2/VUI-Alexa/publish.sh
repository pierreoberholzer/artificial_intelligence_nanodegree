#!/usr/bin/env bash

rm index.zip
cd src
zip -r -X ../index.zip *
cd ..
aws lambda update-function-code --function-name myHistoryFunction --zip-file fileb://index.zip