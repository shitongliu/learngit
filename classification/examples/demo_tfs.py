import grpc
import os
import time
import numpy as np
from tensorflow import make_tensor_proto
from tensorflow.io import read_file
from tensorflow.image import decode_jpeg, resize
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / sum(exp_x)


def request_server(img_resized, server_url):
    '''
    用于向TensorFlow Serving服务请求推理结果的函数。
    :param img_resized: 经过预处理的待推理图片数组，numpy array，shape：(h, w, 3)
    :param server_url: TensorFlow Serving的地址加端口，str，如：'0.0.0.0:8500'
    :return: 模型返回的结果数组，numpy array
    '''
    # Request.
    channel = grpc.insecure_channel(server_url)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "cnn3"  # 模型名称，启动容器命令的model_name参数
    request.model_spec.signature_name = "serving_default"  # 签名名称
    # "conv2d_input"是导出模型时设置的输入名称
    request.inputs["conv2d_input"].CopyFrom(
        make_tensor_proto(img_resized, shape=[1, ] + list(img_resized.shape)))
    response = stub.Predict(request, 5.0)  # 5 secs timeout
    return np.asarray(response.outputs["dense_1"].float_val)    # 输出名称


if __name__ == '__main__':
    # load class names
    class_names = []
    for cn in os.listdir(r'..\data\test'):
        class_names.append(cn)

    # load image tensor
    img_raw = read_file(r'..\data\predict\20200806_IMG_1116.JPG')
    img_tensor = decode_jpeg(img_raw, channels=3)
    img_tensor = resize(img_tensor, (256, 256)) / 255

    # send request
    url = '192.168.56.101:8500'
    tic = time.time()
    prob = softmax(request_server(img_tensor, url))
    toc = time.time()
    ans = class_names[int(np.argmax(prob))]

    # print result
    print('{0:22}{1}'.format('class name', 'probability'))
    for n, p in zip(class_names, prob):
        n = '***' + n + '***' if n == ans else n
        print('{0:22}{1:4.2f}%'.format(n, p*100))

    print('\nTime cost: {0:.4f}s'.format(toc - tic))
    print('done.')

