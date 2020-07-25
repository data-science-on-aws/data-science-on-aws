# Welcome to your CDK TypeScript project!

This is a blank project for TypeScript development with CDK.

The `cdk.json` file tells the CDK Toolkit how to execute your app.

## Settings

Make sure you change [setup](./bin/cdk-setup.ts) REPLACE_ME with the account id and user name you want for the admin account.
If you already have a user, specify the name of the user and set createUser to false
If you don't want to use explicit access policies set explicitAccessPolicy to false

## Useful commands

 * `npm run build`   compile typescript to js
 * `npm run watch`   watch for changes and compile
 * `npm run test`    perform the jest unit tests
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk synth`       emits the synthesized CloudFormation template

* `npm run clean`   clean the output js files after making changes to the stack