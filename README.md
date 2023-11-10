ゼロから作る Deep Learning ❺
=============================

[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-5/images/deep-learning-from-scratch-5.png" width="200px">](https://www.amazon.co.jp/dp/4814400594/)


書籍『[ゼロから作るDeep Learning ❺](https://www.amazon.co.jp/dp/4814400594/)』（オライリー・ジャパン発行）のサポートサイトです。本書籍で使用するソースコードがまとめられています。


## ファイル構成

|フォルダ名 |説明                             |
|:--        |:--                              |
|`step01`   |ステップ1で使用するソースコード  |
|`step02`   |ステップ2で使用するソースコード  |
|...        |...                              |
|`step10`   |ステップ10で使用するソースコード |


ソースコードの解説は本書籍をご覧ください。


## Pythonと外部ライブラリ

ソースコードを実行するには下記のライブラリが必要です。

* NumPy
* Matplotlib
* PyTorch（バージョン：2.x）
* torchvision
* tqdm

※Pythonのバージョンは **3系** を利用します。


## 実行方法

各章のフォルダへ移動して、Pythonコマンドを実行します。

```
$ cd step01
$ python norm_dist.py

$ cd ../step02
$ python generate.py
```


<!--
## クラウドサービスでの実行

本書のコードは次の表にあるボタンをクリックすることで、AWSの無料の計算環境である[Amazon SageMaker Studio Lab](https://studiolab.sagemaker.aws/)上に実行できます(事前に[メールアドレスによる登録](https://studiolab.sagemaker.aws/requestAccount)が必要です)。SageMaker Studio Labの使い方は[こちら](https://github.com/aws-sagemaker-jp/awesome-studio-lab-jp/blob/main/README_usage.md)をご覧ください。[Amazon SageMaker Studio Lab Community](https://github.com/aws-studiolab-jp/awesome-studio-lab-jp)で最新情報が取得できます。

|フォルダ名 |Amazon SageMaker Studio Lab
|:--        |:--                          |
|step01       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/step01.ipynb)|
|step02       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/step02.ipynb)|
|step03       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/step03.ipynb)|
|step04       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/step04.ipynb)|
|step05       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/step05.ipynb)|
|step06       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/step06.ipynb)|
|step07       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/step07.ipynb)|
|step08       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/step08.ipynb)|
|step09       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/step09.ipynb)|
|step10       |[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/oreilly-japan/deep-learning-from-scratch-5/blob/master/notebooks/step10.ipynb)|
 -->


## ライセンス

本リポジトリのソースコードは[MITライセンス](http://www.opensource.org/licenses/MIT)です。
商用・非商用問わず、自由にご利用ください。


## 正誤表

本書の正誤情報は以下のページで公開しています。

https://github.com/oreilly-japan/deep-learning-from-scratch-5/wiki/errata

本ページに掲載されていない誤植など間違いを見つけた方は、[japan@oreilly.co.jp](<mailto:japan@oreilly.co.jp>)までお知らせください。