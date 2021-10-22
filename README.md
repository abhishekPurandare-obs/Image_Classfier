In the master branch. We created a simple experiment that we can run from any system as long as MLflow is able to find `MLProject` file.

In this section, we will cover how to train the model and create an endpoint using AWS SageMaker.


# Contents

1. [Setting up the tracking server on EC2](#1-setting-up-the-tracking-server-on-ec2)
2. [Packaging the source code for training on sagemaker](#2-containerizing-the-source-code-for-training-on-sagemaker)
3. [Training your model on sagemaker](#3-training-your-model-on-sagemaker)
4. [Creating an endpoint using MLflow sagemaker API](#4-creating-an-endpoint-using-mlflow-sagemaker-api)

## 1. Setting up the tracking server on EC2

In our local runs, by default the MLflow tracking server runs on http://0.0.0.0:5000/ and artifacts are saved in project's root directory.

When tracking our experiments we need to host this tracking server publicly. Each run will create a set of artifacts like files, model weights, etc as well as metrics and logs. To store all metrics and logs, mlflow allows us to use a database and to store artifacts, an S3 bucket.

I won't go in details on how to set this up. But simply create an S3 bucket and a Database in AWS RDS. In this case, I've used PostgreSQL. Once that's done refer to [project.env](project.env) file where I've set up Database credentials and the bucket name. Make sure you create an S3 bucket with Read/Write access.

Next, it's time to set up our server on EC2. For this we won't be needing a high end machine. We can use `t2.micro` which is a free-tier instance. While configuring this instance, make sure to provide sufficient amount of memory, so that our image can be stored. And to make the server publicly accessible, you'll have to set security groups with inbound and outbound rules for public access or access specific to your machines. Once that is set, do the following,

1. log in to your instance through `ssh`
2. update your system
3. Install docker (refer to the official docker documentation).
4. Transfer your `project.env` and `Dockerfile.tracker` using `scp` to EC2 instance.
5. Build your image on the instance itself `docker build -t mlflow-tracker -f Dockerfile.tracker .`
6. Run your server `docker run --detach --env-file project.env -t mlflow-tracker:latest`
7. Visit the dashboard on your tracking uri.

If you are unable to load the page, it means you have not configured your security groups properly. In this particular case, I have allowed all traffic to and from my EC2 instance. But try to avoid this apporach and allow access only to specific IP addresses.

Now test your server out locally by running `mlflow run` command that will parse `MLProject` file. If your run is successful. All your artifacts and metrics will be stored in the bucket and database. You'll be able to access this information from mlflow dashboard running on tracking server.



## 2. Containerizing the source code for training on sagemaker

AWS Sagemaker is a well-equipped tool when it comes to end-to-end machine learning. Most often ML engineers will be using Sagemaker's Built-in Algorithms on their custom datasets. But SageMaker also provides us a way to bring our own model. To execute our custom training scripts, SageMaker provides us a script mode where we can train our model in a container. Frameworks such as TensorFlow, PyTorch, MXNet, etc. are provided in these containers. But sometimes we may need additional dependencies which may not be available like mlflow, for instance. Therefore we bring in our own container.

In this section, we will go through how to create this custom container and how to push our image to ECR.

To start with, we need an image with all the libraries installed. `Dockerfile.mlflowbase` installs all the modules and packages that we need. We take this as a base image and then put all the source code, environment variables, etc on top of it. The reason why I'm creating a base image beforehand is, because installing modules takes some time and if we want to change the value of one of these environment variables we'll need to rebuild with changes because docker will reinstall all the packages. Unfortunately, there is no way to pass your docker arguments from sagemaker. So something like a .env file could have brought some flexibility. But at this point, I have not found a way to do this or at least there is none.

In `Dockerfile`,

```
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
```

for disabling python buffering stdout and not writing .pyc files in our containers.

`ENV PATH="/opt/program:${PATH}"` adding the workdir path to `PATH` variable so that sagemake will be able to find our training script.


```
ENV MLFLOW_TRACKING_URI=http://ec2-13-232-139-2.ap-south-1.compute.amazonaws.com:5000/
ENV MLFLOW_EXP_NAME=Experiment-abhishek
```

We set these env variables to let mlflow know the tracking server uri and the name of our experiments, we had already created our tracking server in the first section.


`train` is the training script which is called by sagemaker. 
```
RUN chmod +x train
```
Make sure to make this script executable otherwise sagemaker won't be able to run it. This training script takes some parameters that will be passed from sagemaker notebook instance. Two of the parameters are already set using env variables. The tracking server uri and the experiment name. I have added these two just in case if we do want to change. I'm setting the values using mlflow API. You can check the code. This approach brings in the flexibility we need.

Now that our Dockerfile is set, all we need to do is build and push our image to AWS ECR so that it can be accessed by sagemaker.

`docker build -t mlflow-image:latest -f Dockerfile .`


First create a repository, in your ECS repositories. from there you click on `view push commands` and see how to push the image or you can just set up [this](https://github.com/aws/amazon-sagemaker-examples/blob/master/advanced_functionality/scikit_bring_your_own/container/build_and_push.sh) script.



## 3. Training your model on sagemaker

1. Go to AWS SageMaker dashboard and create a notebook instance
    - select `ml.t2.medium` for your notebook instance. Other instances will be more costly.
    - Setup an existing execution role for the notebook. Make sure you have provided S3 as well as ECR access to this role. Otherwise sagemaker won't be able to pull the image or store its artifacts to S3.
    - If needed, set up your VPC, subnets and security groups.
    - If needed, set up a git repository that will be pulled in the same notebook instance. I haven't done this because our code is already inside the container. You can use this if you are going to use existing containers to run a custom script.
    - This step will take some time.
2. Create a notebook with your preffered environment. I chose `conda_tensorflow2_p36`.
3. Refer to [mlflow-train-deploy.ipynb](mlflow-train-deploy.ipynb) to see how the model is trained.
    - Specify your image uri which is stored in ECR
    - Specify your hyperparameters. These values are passed to your training script as command line arguments.
    - Specify output path. This is used by the sagemaker to save your model as tar file. Although we won't be needing it, this argument is mandatory.
    - Specify your execution role. You can either get existing role by calling `sagemaker.Session.get_execution_role()` or just specify arn to your role from IAM.
    - I have created two sets of parameters for training. One is `local` mode where the training will happen on the notebook instance itself. It would be a good practice to test your estimator fit using this mode to make sure everything is working as expected.

    - In second set of parameters, I have specified `ml.m4.xlarge` as well as other parameters such as 
    ```
    'use_spot_instances': True,
    'max_run': 500,
    'max_wait': 600
    ``` 

    explain these

    - You simply call fit method on your estimator.


Now in my case. I copied my dataset inside a container since it wasn't too big. But a good practice would be to store the dataset in a bucket and pass your train and test paths to your training script. The `estimator.fit()` in the notebook can take these paths as inputs.

## 4. Creating an endpoint using MLflow sagemaker API