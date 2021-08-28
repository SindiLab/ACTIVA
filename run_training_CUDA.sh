CUDA_VISIBLE_DEVICES=0 python ACTIVA.py --m_plus 110 --num_vae 10 --num_cf 5 --nEpochs 500 

## to run on multiple GPUs
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ACTIVA.py --m_plus 110 --num_vae 10 --num_cf 5 --nEpochs 500 

## helpful tip to AWS users: to stop running AWS right after running the code
# aws ec2 stop-instances --instance-ids $INSTANCE_ID --hibernate --region us-west-2
