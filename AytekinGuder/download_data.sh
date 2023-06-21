# put your kaggle credentials here
export KAGGLE_USERNAME="" KAGGLE_KEY=""

kaggle datasets download ambityga/imagenet100
unzip imagenet100.zip -d imagenet100
mkdir imagenet100/train && mv imagenet100/train.X*/* imagenet100/train/
rm -rf imagenet100/train.X*
mv imagenet100/val.X imagenet100/val