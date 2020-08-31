# NeurIPS 2020 - Procgen Starter Kit with Amazon SageMaker Reinforcement Learning
This is the Amazon SageMaker Reinforcement Learning starter kit for the [NeurIPS 2020 - Procgen competition](https://www.aicrowd.com/challenges/neurips-2020-procgen-competition) hosted on [AIcrowd](https://www.aicrowd.com/).
​
Amazon SageMaker is a fully managed service that enables you to build and deploy models faster and with less heavy lifting. Amazon SageMaker has built-in features to assist with data labeling and preparation; training, tuning and debugging models; and deploying and monitoring models in production. This notebook uses the fully managed RL capabilities in Amazon SageMaker, which include pre-packaged RL toolkits and fully managed model training and deployment and builds on top of the algorithms and libraries of the [NeurIPS 2020 - Procgen competition](https://www.aicrowd.com/challenges/neurips-2020-procgen-competition) hosted on [AIcrowd](https://www.aicrowd.com/) Additionally, Amazon SageMaker Managed Spot Training is used to reduce training costs by up to 90%. 
​
For more information, see Amazon SageMaker Experiments – Organize, Track And Compare Your Machine Learning Trainings. For more information about applying RL to domains such as recommendation systems, robotics, financial management, and more, see the [GitHub repo](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/reinforcement_learning).
​
# ��️ About ProcGen Benchmark
​
16 simple-to-use procedurally-generated [gym](https://github.com/openai/gym) environments which provide a direct measure of how quickly a reinforcement learning agent learns generalizable skills.  The environments run at high speed (thousands of steps per second) on a single core.
​
![](https://raw.githubusercontent.com/openai/procgen/master/screenshots/procgen.gif)
​
These environments are associated with the paper [Leveraging Procedural Generation to Benchmark Reinforcement Learning](https://cdn.openai.com/procgen.pdf) [(citation)](#citation). Compared to [Gym Retro](https://github.com/openai/retro), these environments are:
​
* Faster: Gym Retro environments are already fast, but Procgen environments can run >4x faster.
* Non-deterministic: Gym Retro environments are always the same, so you can memorize a sequence of actions that will get the highest reward.  Procgen environments are randomized so this is not possible.
* Customizable: If you install from source, you can perform experiments where you change the environments, or build your own environments.  The environment-specific code for each environment is often less than 300 lines.  This is almost impossible with Gym Retro.
​
​
# �� Getting Started
​
​
### Get an AWS account
​
You will need an AWS account to use this solution. Sign up for an account here (https://aws.amazon.com/).
You will also need to have permission to use AWS CloudFormation (https://aws.amazon.com/cloudformation/) and to create all the resources detailed in the architecture section (https://github.com/awslabs/aws-fleet-predictive-maintenance/#architecture). All AWS permissions can be managed through AWS IAM (https://aws.amazon.com/iam/). Admin users will have the required permissions, but please contact your account's AWS administrator if your user account doesn't have the required permissions.
​
### Launch the solution
​
Click on one of the following buttons to quick create the AWS CloudFormation Stack:
​
<table>
  <tr>
    <th colspan="3">AWS Region</td>
    <th>AWS CloudFormation</td>
  </tr>
  <tr>
    <td>US West</td>
    <td>Oregon</td>
    <td>us-west-2</td>
    <td align="center">
      <a href="https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/create/review?templateURL=https://sagemaker-solutions-us-west-2.s3-us-west-2.amazonaws.com/rl-procgen-neurips/cloudformation/sagemaker.yaml&stackName=sagemaker-solutions-rl-procgen-neuips">
        <img src="docs/launch_button.svg" height="30">
      </a>
    </td>
  </tr>
</table>
​
You should acknowledge the use of the two capabilities and click 'Create Stack'. Once stack creation has completed successfully, you could start training the model with `train.ipynb`.

# �� Submission [Same as in NeurIPS 2020 - Procgen competition]
​
Same as in [NeurIPS 2020 - Procgen competition](https://www.aicrowd.com/challenges/neurips-2020-procgen-competition) hosted on [AIcrowd](https://www.aicrowd.com/).
​
Happy Submitting!! :rocket:
​
​
# Author(s)
- [Jonathan Chung](https://github.com/jonomon)
- [Anna Luo]()
- [Yunzhe Tao]()
- [Sahika Genc]()
- [Sharada Mohanty](https://twitter.com/MeMohanty/)
- [Karl Cobbe](https://github.com/kcobbe)
- [Jyotish](https://github.com/jyotishp)
- [Shivam Khandelwal](https://github.com/skbly7)


## License

This project is licensed under the Apache-2.0 License.
