import boto3
import csv
import json

CLASSES = {'NAO': '0', 'CCD': '1', 'CCE': '2', 'MLOD': '3', 'MLOE': '4'}

def lambda_handler(event, context):
    print('starting...') 
    
    s3 = boto3.resource('s3')
    
    #bucket containing the immages
    images_bucket='<<images_bucket_name>>'
    
    #bucket to write the lst files 
    output_bucket='<<output_bucket_name>>'
    
    prefix = 'resize'
    train_folder = 'train'
    test_folder = 'test'    
    train_file_name = 'train-data.lst'
    test_file_name = 'test-data.lst'
    
    s3train = '{}/{}/'.format(prefix, train_folder)
    s3validation = '{}/{}/'.format(prefix, test_folder)
    
    s3train_lst = '{}/{}'.format(prefix, train_file_name)
    s3test_lst = '{}/{}'.format(prefix, test_file_name)
    
    my_images_bucket = s3.Bucket(images_bucket)
    
    #generate a file containing the paths for the training files
    print('generating file {}'.format(s3train_lst)) 
    with open('/tmp/train-data.lst', 'w', newline='') as file:
        writer = csv.writer(file, delimiter = '\t')
        cont = 0
        for object_summary in my_images_bucket.objects.filter(Prefix=s3train):
            s = object_summary.key
            if s.endswith("jpg") or s.endswith("jpeg") or s.endswith("bmp"):
                cont += 1
                ss = s[len(s3train)::]
                k = str(ss.split('/')[0])
                writer.writerow([cont, CLASSES.get(k), ss])
    
    #generate a file containing the paths for the test files
    print('generating file {}'.format(s3test_lst)) 
    with open('/tmp/test-data.lst', 'w', newline='') as file:
        writer = csv.writer(file, delimiter = '\t')
        cont = 0
        for object_summary in my_images_bucket.objects.filter(Prefix=s3validation):
            s = object_summary.key
            if s.endswith("jpg") or s.endswith("jpeg") or s.endswith("bmp"):
                cont += 1
                ss = s[len(s3validation)::]
                k = str(ss.split('/')[0])
                writer.writerow([cont, CLASSES.get(k), ss])
    
    print('writing file {} to S3'.format(s3train_lst))         
    s3.meta.client.upload_file('/tmp/train-data.lst', output_bucket, s3train_lst)
    
    print('writing file {} to S3'.format(s3test_lst)) 
    s3.meta.client.upload_file('/tmp/test-data.lst', output_bucket, s3test_lst)
    
    print('done!')
    
    return {
        'message': 'OK'
    }