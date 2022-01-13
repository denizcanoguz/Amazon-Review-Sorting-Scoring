import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Rating Product &
# Sorting Reviews in Amazon


# İş Problemi
# ---------------------------------------------------------
# Ürün ratinglerini daha doğru hesaplamaya
# çalışmak ve ürün yorumlarını daha doğru
# sıralamak.

# Veri Seti Hikayesi
# ---------------------------------------------------------
# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile
# çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı
# puanları ve yorumları vardır.

#Değişkenler
# ---------------------------------------------------------
# reviewerID – Kullanıcı ID’si
# Örn: A2SUAM1J3GNN3B

# asin – Ürün ID’si.
# Örn: 0000013714

# reviewerName – Kullanıcı Adı
# reviewText – Değerlendirme

# helpful – Faydalı değerlendirme derecesi
# Örn: 2/3

# reviewText – Değerlendirme
# Kullanıcının yazdığı inceleme metni

# summary – Değerlendirme özeti

# overall – Ürün rating’i

# reviewTime – Değerlendirme zamanı
# Raw

# unixReviewTime – Değerlendirme zamanı
# Unix time

# day_diff – Değerlendirmeden itibaren geçen gün sayısı

# helpful_yes – Değerlendirmenin faydalı bulunma sayısı

# total_vote – Değerlendirmeye verilen oy sayısı

# Proje Aşamaları
# ---------------------------------------------------------

df = pd.read_csv("datasets/amazon_review.csv")
df.head()

df.info()
# Görev 1:
# Average Rating’i güncel yorumlara göre
# hesaplayınız ve var olan average rating ile
# kıyaslayınız.
df["overall"].mean()  # Genel Puan Ortalaaması

df.loc[df["day_diff"] <= 30, "overall"].mean() * 28 / 100 + \
df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * 26 / 100 + \
df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * 24 / 100 + \
df.loc[(df["day_diff"] > 180), "overall"].mean() * 22 / 100


# Görev 2:
# Ürün için ürün detay sayfasında
# görüntülenecek 20 review’i belirleyiniz.
df["helpful_no"] = df["total_vote"]-df["helpful_yes"]  # Faydalı olmayan yorumlar

comments = pd.DataFrame({"up": df["helpful_yes"], "down": df["helpful_no"]})


def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"],x["down"]),axis=1)

comments.sort_values("wilson_lower_bound", ascending=False).head(20)


