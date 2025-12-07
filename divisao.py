# # import os
# # import shutil
# # import random
# # import cv2

# # # Pasta original com as classes (1,2,3,4,5,6)
# # source_dir = "Nodules"

# # # Pastas destino
# # train_dir = "dataset/train"
# # val_dir = "dataset/val"
# # test_dir = "dataset/test"

# # # -------- 1. LIMPAR AS PASTAS ANTES DE CRIAR --------
# # for d in [train_dir, val_dir, test_dir]:
# #     if os.path.exists(d):
# #         shutil.rmtree(d)
# #     os.makedirs(d)

# # # Proporções
# # train_ratio = 0.7
# # test_ratio = 0.2
# # val_ratio = 0.1

# # # -------- 2. DIVISÃO DAS IMAGENS --------
# # for class_name in sorted(os.listdir(source_dir)):

# #     class_path = os.path.join(source_dir, class_name)
# #     if not os.path.isdir(class_path):
# #         continue

# #     images = sorted(os.listdir(class_path))
# #     random.shuffle(images)

# #     n = len(images)
# #     n_train = int(n * train_ratio)
# #     n_test = int(n * test_ratio)
# #     n_val = n - n_train - n_test

# #     # Cria subpastas
# #     os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
# #     os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
# #     os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

# #     # Divisão
# #     train_imgs = images[:n_train]
# #     test_imgs = images[n_train:n_train + n_test]
# #     val_imgs = images[n_train + n_test:]

# #     # Copiar arquivos
# #     for img in train_imgs:
# #         shutil.copy2(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

# #     for img in test_imgs:
# #         shutil.copy2(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

# #     for img in val_imgs:
# #         shutil.copy2(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))


# # # -------- 3. BALANCEAMENTO DAS CLASSES NO TRAIN --------

# # def rotate_image(img, angle):
# #     return cv2.rotate(img, angle)

# # print("\nGerando imagens para balanceamento...")

# # # Conta as imagens do train
# # counts = {}
# # for cls in sorted(os.listdir(train_dir)):
# #     cls_path = os.path.join(train_dir, cls)
# #     counts[cls] = len(os.listdir(cls_path))

# # max_count = max(counts.values())

# # print(f"\nMaior classe tem {max_count} imagens\n")

# # for cls, count in counts.items():
# #     if count == max_count:
# #         continue

# #     cls_path = os.path.join(train_dir, cls)
# #     imgs = os.listdir(cls_path)

# #     print(f"Balanceando Classe {cls} ({count} → {max_count})")

# #     idx = 0
# #     while len(os.listdir(cls_path)) < max_count:

# #         img_name = imgs[idx % len(imgs)]
# #         img_path = os.path.join(cls_path, img_name)

# #         img = cv2.imread(img_path)
# #         if img is None:
# #             idx += 1
# #             continue

# #         # Rotaciona (90°, 180°, 270°)
# #         angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

# #         for a in angles:
# #             new_img = rotate_image(img, a)

# #             base = img_name.rsplit(".", 1)[0]
# #             new_name = f"{base}_rot{a}_{random.randint(1000,9999)}.png"

# #             cv2.imwrite(os.path.join(cls_path, new_name), new_img)

# #             if len(os.listdir(cls_path)) >= max_count:
# #                 break

# #         idx += 1


# # # -------- 4. MOSTRAR RESUMO FINAL --------
# # def contar(split_dir):
# #     print(f"\n{split_dir}:")
# #     for class_name in sorted(os.listdir(split_dir)):
# #         class_path = os.path.join(split_dir, class_name)
# #         print(f"Classe {class_name}: {len(os.listdir(class_path))} imagens")

# # print("\n======= RESUMO FINAL =======")
# # contar(train_dir)
# # contar(test_dir)
# # contar(val_dir)

# # import os
# # import shutil
# # import random
# # import cv2
# # import numpy as np

# # # ========= CONFIG =========
# # source_dir = "Nodules"

# # train_dir = "dataset/train"
# # val_dir = "dataset/val"
# # test_dir = "dataset/test"

# # CLASSES = ["1", "2", "3"]

# # train_ratio = 0.7
# # test_ratio = 0.2
# # val_ratio = 0.1

# # # ========= FUNÇÕES DE AUGMENTAÇÃO =========
# # def rotate(img):
# #     angles = [
# #         cv2.ROTATE_90_CLOCKWISE,
# #         cv2.ROTATE_180,
# #         cv2.ROTATE_90_COUNTERCLOCKWISE
# #     ]
# #     return cv2.rotate(img, random.choice(angles))

# # def flip(img):
# #     return cv2.flip(img, random.choice([0, 1]))

# # # Agora só usa rotate e flip
# # AUG_FUNCS = [rotate, flip]


# # # ========= 1. BALANCEAR AS CLASSES NO PRÓPRIO "Nodules" =========
# # print("\n===== BALANCEANDO AS CLASSES =====\n")

# # # descobrir maior classe
# # counts = {}
# # for cls in CLASSES:
# #     cls_path = os.path.join(source_dir, cls)
# #     counts[cls] = len(os.listdir(cls_path))

# # max_count = max(counts.values())
# # print(f"Maior classe = {max_count} imagens\n")

# # # aumentar cada classe menor
# # for cls in CLASSES:
# #     cls_path = os.path.join(source_dir, cls)
# #     imgs = os.listdir(cls_path)
# #     initial_count = len(imgs)

# #     print(f"Classe {cls}: {initial_count} → {max_count}")

# #     idx = 0
# #     while len(os.listdir(cls_path)) < max_count:
# #         img_name = imgs[idx % len(imgs)]
# #         img_path = os.path.join(cls_path, img_name)

# #         img = cv2.imread(img_path)
# #         if img is None:
# #             idx += 1
# #             continue

# #         func = random.choice(AUG_FUNCS)
# #         new_img = func(img)

# #         base = img_name.rsplit(".", 1)[0]
# #         new_name = f"{base}_aug_{random.randint(100000,999999)}.png"
# #         cv2.imwrite(os.path.join(cls_path, new_name), new_img)

# #         idx += 1


# # # ========= 2. LIMPAR E RECRIAR dataset/train val test =========
# # for d in [train_dir, val_dir, test_dir]:
# #     if os.path.exists(d):
# #         shutil.rmtree(d)
# #     os.makedirs(d)

# # for cls in CLASSES:
# #     os.makedirs(os.path.join(train_dir, cls))
# #     os.makedirs(os.path.join(val_dir, cls))
# #     os.makedirs(os.path.join(test_dir, cls))


# # # ========= 3. FAZER O SPLIT AGORA QUE JÁ ESTÁ BALANCEADO =========
# # print("\n===== REALIZANDO SPLIT =====\n")

# # for cls in CLASSES:

# #     cls_path = os.path.join(source_dir, cls)

# #     images = sorted(os.listdir(cls_path))
# #     random.shuffle(images)

# #     n = len(images)
# #     n_train = int(n * train_ratio)
# #     n_test = int(n * test_ratio)
# #     n_val = n - n_train - n_test

# #     train_imgs = images[:n_train]
# #     test_imgs  = images[n_train:n_train+n_test]
# #     val_imgs   = images[n_train+n_test:]

# #     for img in train_imgs:
# #         shutil.copy2(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))

# #     for img in test_imgs:
# #         shutil.copy2(os.path.join(cls_path, img), os.path.join(test_dir, cls, img))

# #     for img in val_imgs:
# #         shutil.copy2(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))

# #     print(f"Classe {cls}: TRAIN={len(train_imgs)}, VAL={len(val_imgs)}, TEST={len(test_imgs)}")


# # # ========= 4. RESUMO =========
# # print("\n===== RESUMO FINAL =====")

# # def contar(path):
# #     print(f"\n{path}:")
# #     for c in CLASSES:
# #         p = os.path.join(path, c)
# #         print(f"Classe {c}: {len(os.listdir(p))} imagens")

# # contar(train_dir)
# # contar(val_dir)
# # contar(test_dir)
# # import os
# # import shutil
# # import random
# # import cv2
# # import numpy as np

# # # ========= CONFIG =========
# # source_dir = "Nodules"

# # train_dir = "dataset/train"
# # val_dir   = "dataset/val"
# # test_dir  = "dataset/test"

# # CLASSES = ["1", "2", "3"]

# # train_ratio = 0.7
# # test_ratio  = 0.2
# # val_ratio   = 0.1


# # # ========= FUNÇÕES DE AUGMENTAÇÃO =========
# # def rotate_custom(img):
# #     angles = [90, 135, 180]
# #     angle = random.choice(angles)

# #     h, w = img.shape[:2]
# #     center = (w // 2, h // 2)

# #     M = cv2.getRotationMatrix2D(center, angle, 1.0)
# #     return cv2.warpAffine(img, M, (w, h))

# # def flip(img):
# #     return cv2.flip(img, random.choice([0, 1]))

# # AUG_FUNCS = [rotate_custom, flip]


# # # ========= 1. LIMPAR E RECRIAR dataset/train-val-test =========
# # for d in [train_dir, val_dir, test_dir]:
# #     if os.path.exists(d):
# #         shutil.rmtree(d)
# #     os.makedirs(d)

# # for cls in CLASSES:
# #     os.makedirs(os.path.join(train_dir, cls))
# #     os.makedirs(os.path.join(val_dir, cls))
# #     os.makedirs(os.path.join(test_dir, cls))


# # # ========= 2. SPLIT INICIAL SEM AUGMENTAR =========
# # print("\n===== REALIZANDO SPLIT (ORIGINAL) =====\n")

# # for cls in CLASSES:
# #     cls_path = os.path.join(source_dir, cls)
# #     images = sorted(os.listdir(cls_path))
# #     random.shuffle(images)

# #     n = len(images)
# #     n_train = int(n * train_ratio)
# #     n_test  = int(n * test_ratio)
# #     n_val   = n - n_train - n_test

# #     train_imgs = images[:n_train]
# #     test_imgs  = images[n_train:n_train+n_test]
# #     val_imgs   = images[n_train+n_test:]

# #     # copiar
# #     for img in train_imgs:
# #         shutil.copy2(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))

# #     for img in test_imgs:
# #         shutil.copy2(os.path.join(cls_path, img), os.path.join(test_dir, cls, img))

# #     for img in val_imgs:
# #         shutil.copy2(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))

# #     print(f"Classe {cls}: TRAIN={len(train_imgs)}, VAL={len(val_imgs)}, TEST={len(test_imgs)}")


# # # ========= 3. BALANCEAR APENAS O TREINO =========
# # print("\n===== AUMENTANDO SOMENTE O TREINO =====\n")

# # # descobrir maior classe no treino
# # counts = {}
# # for cls in CLASSES:
# #     cls_path = os.path.join(train_dir, cls)
# #     counts[cls] = len(os.listdir(cls_path))

# # max_count = max(counts.values())
# # print(f"Maior classe no treino = {max_count}\n")

# # # augment apenas nas classes menores
# # for cls in CLASSES:
# #     cls_path = os.path.join(train_dir, cls)
# #     imgs = os.listdir(cls_path)
    
# #     print(f"Classe {cls}: {len(imgs)} → {max_count}")

# #     idx = 0
# #     while len(os.listdir(cls_path)) < max_count:
# #         img_name = imgs[idx % len(imgs)]
# #         img_path = os.path.join(cls_path, img_name)

# #         img = cv2.imread(img_path)
# #         if img is None:
# #             idx += 1
# #             continue

# #         func = random.choice(AUG_FUNCS)
# #         new_img = func(img)

# #         base = img_name.rsplit(".", 1)[0]
# #         new_name = f"{base}_aug_{random.randint(100000,999999)}.png"
# #         cv2.imwrite(os.path.join(cls_path, new_name), new_img)

# #         idx += 1


# # # ========= 4. RESUMO FINAL =========
# # print("\n===== RESUMO FINAL =====")

# # def contar(path):
# #     print(f"\n{path}:")
# #     for c in CLASSES:
# #         p = os.path.join(path, c)
# #         print(f"Classe {c}: {len(os.listdir(p))} imagens")

# # contar(train_dir)
# # contar(val_dir)
# # contar(test_dir)
# import os
# import shutil
# import random

# # ========= CONFIG =========
# source_dir = "Nodules"

# train_dir = "dataset/train"
# val_dir   = "dataset/val"
# test_dir  = "dataset/test"

# CLASSES = ["1", "2", "3"]

# train_ratio = 0.7
# test_ratio  = 0.2
# val_ratio   = 0.1


# # ========= 1. LIMPAR E RECRIAR dataset/train-val-test =========
# for d in [train_dir, val_dir, test_dir]:
#     if os.path.exists(d):
#         shutil.rmtree(d)
#     os.makedirs(d)

# for cls in CLASSES:
#     os.makedirs(os.path.join(train_dir, cls))
#     os.makedirs(os.path.join(val_dir, cls))
#     os.makedirs(os.path.join(test_dir, cls))


# # ========= 2. SPLIT SEM AUGMENTAÇÃO =========
# print("\n===== REALIZANDO SPLIT (SEM AUMENTAÇÃO) =====\n")

# for cls in CLASSES:
#     cls_path = os.path.join(source_dir, cls)
#     images = sorted(os.listdir(cls_path))
#     random.shuffle(images)

#     n = len(images)
#     n_train = int(n * train_ratio)
#     n_test  = int(n * test_ratio)
#     n_val   = n - n_train - n_test  # resto

#     train_imgs = images[:n_train]
#     test_imgs  = images[n_train:n_train+n_test]
#     val_imgs   = images[n_train+n_test:]

#     # copiar
#     for img in train_imgs:
#         shutil.copy2(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))

#     for img in test_imgs:
#         shutil.copy2(os.path.join(cls_path, img), os.path.join(test_dir, cls, img))

#     for img in val_imgs:
#         shutil.copy2(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))

#     print(f"Classe {cls}: TRAIN={len(train_imgs)}, VAL={len(val_imgs)}, TEST={len(test_imgs)}")


# # ========= 3. RESUMO FINAL =========
# print("\n===== RESUMO FINAL =====")

# def contar(path):
#     print(f"\n{path}:")
#     for c in CLASSES:
#         p = os.path.join(path, c)
#         print(f"Classe {c}: {len(os.listdir(p))} imagens")

# contar(train_dir)
# contar(val_dir)
# contar(test_dir)
import os

base_path = "/home/joyzinhw/classification_CPCNP/dataset/fatias"

extensoes_imagem = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

pastas = sorted([p for p in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, p))])

for pasta in pastas:
    caminho_pasta = os.path.join(base_path, pasta)
    imagens = [
        f for f in os.listdir(caminho_pasta)
        if f.lower().endswith(extensoes_imagem)
    ]
    print(f"Pasta {pasta}: {len(imagens)} imagens")
