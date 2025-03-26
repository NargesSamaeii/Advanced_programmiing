import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import os
import random

# ✅ تنظیم مسیر صحیح داده‌ها
img_dir = r"C:\Users\KARNOO\coco_data\val2017\val2017"  # مسیر جدید تصاویر
ann_file = r"C:\Users\KARNOO\coco_data\annotations\instances_val2017.json"  # مسیر جدید annotations

# ✅ بررسی وجود فایل‌های ضروری
if not os.path.exists(ann_file):
    raise FileNotFoundError(f"⚠ فایل annotations پیدا نشد: {ann_file}")

if not os.path.exists(img_dir):
    raise FileNotFoundError(f"⚠ پوشه تصاویر val2017 پیدا نشد: {img_dir}")

print("✅ مسیرها به درستی تنظیم شده‌اند!")

# 🚀 بارگذاری مجموعه داده COCO
coco = COCO(ann_file)

# 🎯 گرفتن تصاویر مربوط به کلاس خودرو (category_id = 3)
category_id = 3  # کلاس خودرو در COCO
image_ids = coco.getImgIds(catIds=[category_id])
random.shuffle(image_ids)

# 🏗️ ساخت مدل Faster R-CNN
def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 91)  # 91 کلاس COCO
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)
model.train()

# ⚙️ تابع تبدیل تصاویر برای PyTorch
transform = T.Compose([T.ToTensor()])

# 🎓 **آموزش مدل روی داده‌های خودرو**
epochs = 20
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
losses = []

for epoch in range(epochs):
    total_loss = 0
    for img_id in image_ids[:20]:  # آموزش فقط روی 20 تصویر برای تست
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])

        # ✅ بررسی مسیر تصویر قبل از خواندن
        if not os.path.exists(img_path):
            print(f"⚠ تصویر یافت نشد: {img_path}")
            continue

        # 🖼 بارگذاری و تبدیل تصویر
        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠ خطا در لود تصویر: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = transform(image).to(device)

        # 📌 بارگذاری برچسب‌ها و جعبه‌های محدودکننده
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[category_id])
        anns = coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(category_id)

        if len(boxes) == 0:
            print(f"⚠ هیچ خودرویی در این تصویر یافت نشد: {img_path}")
            continue

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
            "labels": torch.tensor(labels, dtype=torch.int64).to(device),
        }

        # ✅ محاسبه Loss و بهینه‌سازی مدل
        loss_dict = model([img_tensor], [target])
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(image_ids[:20])
    losses.append(avg_loss)
    print(f"🎯 Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# **📊 نمایش نمودار تغییرات خطا**
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), losses, marker='o', linestyle='-', color='b', label='Total Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()

# **🚗 تست مدل روی چند تصویر جدید**
model.eval()

def test_model(image_id):
    img_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(img_dir, img_info["file_name"])

    # ✅ بررسی مسیر تصویر
    if not os.path.exists(img_path):
        print(f"⚠ تصویر تست یافت نشد: {img_path}")
        return

    # 🖼 بارگذاری تصویر
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠ خطا در لود تصویر تست: {img_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = transform(image).to(device)

    # 🚀 اجرای مدل روی تصویر
    with torch.no_grad():
        prediction = model([img_tensor])

    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()

    # ✅ نمایش تشخیص‌ها
    threshold = 0.5
    for i in range(len(boxes)):
        if scores[i] > threshold and labels[i] == category_id:
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f"Car: {scores[i]:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Detected Cars - Image ID: {image_id}")
    plt.show()

# ✅ نمایش سه نمونه از تصاویر تستی
for img_id in image_ids[:3]:
    test_model(img_id)