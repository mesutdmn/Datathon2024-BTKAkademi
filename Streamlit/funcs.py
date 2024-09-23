import numpy as np
import pandas as pd
import unicodedata
from transformers import pipeline


pipe = pipeline("text-classification", model="saribasmetehan/bert-base-turkish-sentiment-analysis")
pd.set_option('future.no_silent_downcasting', True)



def get_sentiment(str):
    return pipe(str)[0]["label"]


def normalize_string(s: str) -> str:
    """
    Verilen bir metni aşağıdaki adımlarla normalize eder:

    - Türkçe karakterleri (ör. 'ç', 'ğ', 'ı') Latin alfabesindeki karşılıklarına çevirir.
    - Unicode normalizasyonu (NFKD) kullanarak harflerin üzerindeki aksan işaretlerini kaldırır.
    - Metni küçük harfe çevirir ve baştaki/sondaki boşlukları temizler.
    - Metinden "universitesi" kelimesini kaldırır. (Bazı üniversiteler 2 isimliydi, Örn: Yıldız Teknik Üniversitesi ve Yıldız Teknik)

    Args:
        s (str): Normalize edilecek metin.

    Returns:
        str: Normalize edilmiş metin veya eğer giriş bir string değilse orijinal hali.
    """
    if isinstance(s, str):
        turkish_chars = {
            'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
            'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U'
        }
        s = unicodedata.normalize('NFKD', s)
        s = ''.join([c for c in s if not unicodedata.combining(c)])
        s = ''.join(turkish_chars.get(c, c) for c in s)
        return unicodedata.normalize('NFC', s).lower().replace("universitesi", "").strip()
    return s
def feature_engineering_func(data: pd.DataFrame, is_it_test=False) -> pd.DataFrame :
    
    data = data.copy().map(normalize_string)
    
    data["stk_metni_uzunlugu"] = pd.cut(data["hangi_stknin_uyesisiniz"].str.len(),
                                        bins = [-np.inf,5,50,np.inf],
                                        labels=["az","orta","cok"]).astype("object")
    
    data["girisim_metni_uzunlugu"] = pd.cut(data["girisimcilikle_ilgili_deneyiminizi_aciklayabilir_misiniz"]\
                                            .str.len(), bins = [-np.inf,5,50,np.inf],
                                            labels=["az","orta","cok"]).astype("object")

    burs_turleri = ["kyk","devlet","vakf","koc","meb","bakan","derne","gsb","yurt","tev","tubitak","bos",]
    data["bu_rs_kurumu"] = data["burs_aldigi_baska_kurum"].str.extract(f'({"|".join(burs_turleri)})', expand=False).replace({"yurt":"kyk",
                                                                                                                          "devlet":"kyk",
                                                                                                                          "bakan":"kyk",
                                                                                                                          "gsb":"kyk"})

    lise_turleri = ["fen","imam","saglik","meslek","anadolu","haydarpasa","sosyal",
                    "sanat","spor","acik","kolej","temel","kiz","erkek","ataturk","galatasaray"]
    data["lise_kategorisi"] = data["lise_adi"].str.extract(f'({"|".join(lise_turleri)})', expand=False)
    
    bolum = ["tip","ogretmen","dil","yonetim","sanat","isletme","sosyo","ekonomi","fizyo","cocuk","sosyal","tarih","cografya","odyoloji",
           "muhendis","ecza","hukuk","hemsire","diger","maliye","ebe","istatistik","beslenme","islam","veteriner","plan","yazilim",
             "banka","ticaret","radyo","gazete","felsefe","spor","tercuman","endus","fizik","bilim","peyzaj","turizm","medya","tekno","tasarim",
           "psikoloji","ilahiyat","iktisat","hekim","matematik","mimarlik","biyoloji","kimya","iliski","iletisim"]
        
    data["bölüm_kategorisi"] = data["bölüm"].str.extract(f'({"|".join(bolum)})', expand=False)


    lise_bolumleri = ["sayisal","esit","sozel","sosyal","dil","tm","turkce-matematik",
                      "fen bilimleri","fen", "turkce matematik","bilisim","matematik-fen","fen-matematik","ea","turkce - matematik"]
    
    data["lise_bolumu"] = data["lise_bolumu"].str.extract(f'({"|".join(lise_bolumleri)})', expand=False).replace({"sosyal":"sozel","tm":"esit",
                                                                                                       "fen bilimleri":"sayisal","fen":"sayisal","turkce-matematik":"esit",
                                                                                                       "turkce matematik":"esit","bilisim":"sayisal",
                                                                                                       "matematik-fen":"sayisal","ea":"esit",
                                                                                                       "turkce - matematik":"esit","diger":np.NaN})

    data["lise_turu"] = data["lise_turu"].replace({"meslek":"devlet","ozel":"ozel","devlet":"devlet","diger":"devlet",
                                               "anadolu lisesi":"devlet","meslek lisesi":"devlet",
                                               "ozel lisesi":"ozel","fen lisesi":"ozel","imam hatip lisesi":"devlet","duz lise":"devlet"})
    
    data["universite_kacinci_sinif"] = data["universite_kacinci_sinif"].astype(str).replace({"hazirlik":0}).astype(int)

    data["basvuru_yasi"] = data["yas"]
    data.loc[data["basvuru_yasi"]==26,"basvuru_yasi"] = 25
    data.loc[(data["basvuru_yasi"]<18) | (data["basvuru_yasi"]>25),"basvuru_yasi"] = np.NaN


    top_4 = ["girisimcilik_kulupleri_tarzi_bir_kulube_uye_misiniz",
         "girisimcilikle_ilgili_deneyiminiz_var_mi",
         "aktif_olarak_bir_stk_üyesi_misiniz",
         "profesyonel_bir_spor_daliyla_mesgul_musunuz","ingilizce_biliyor_musunuz"]
        
    data["genel_aktiflik"] = data[top_4].replace({"evet":True,"hayir":False}).sum(axis=1)
    

    data["universite_derecesi"] = data["universite_not_ortalamasi"] * data["universite_kacinci_sinif"].replace({5:4,6:4,0:1}) 
    
    data["is_it_same_sehir"] = data["ikametgah_sehri"] == data["dogum_yeri"]
    data["is_it_same_okul_turu"] = data["universite_turu"] == data["lise_turu"]
    
    #Kişilerin girişim ve stk metinlerinde kullandıkları kelime sayısı sayılarak, karmaşık metinler yazma yetenekleri hesaplandı
    data["girisim_metni_kelime_sayisi"] = data["girisimcilikle_ilgili_deneyiminizi_aciklayabilir_misiniz"].str.split(" ").apply(lambda x: len(x) if isinstance(x, list) else np.NaN)
    data["stk_metni_kelime_sayisi"] = data["hangi_stknin_uyesisiniz"].str.split(" ").apply(lambda x: len(x) if isinstance(x, list) else np.NaN)

    #Kişilerin çeşitli kategorilerdeki ortalamaya göre, notlarının farkı hesaplanarak, ortalamanın neresinde oldukları hesaplandı.
    data["uni_basarisi_digerlerine_gore"] = data.groupby('universite_kacinci_sinif')['universite_not_ortalamasi'].transform(func="mean") - data['universite_not_ortalamasi']
    data["lise_basarisi_digerlerine_gore"] = data.groupby('lise_kategorisi')['lise_mezuniyet_notu'].transform(func="mean") - data['lise_mezuniyet_notu']
    data["bölüm_digerlerine_gore"] = data.groupby('bölüm_kategorisi')['universite_not_ortalamasi'].transform(func="mean") - data['universite_not_ortalamasi']
    
    #Kişiin Liseden sonra üniversitidede puanını koruma durumuna göre ivmesi hesaplandı
    data["basari_ivmesi"] = (data["universite_not_ortalamasi"] * 25 - data["lise_mezuniyet_notu"]) / (data["lise_mezuniyet_notu"] + 0.0001)
    
    #Türkiye İstatistik Kurumundan, Şehirlerde çeşitli kategoriler için eğitimlerde harcanan süre ekstra veri olarak eklendi.
    egitim_suresi = pd.read_csv( "Streamlit/data/egitim_ortalama_train.csv")
    data = pd.merge(data, egitim_suresi, how="left", on="ikametgah_sehri")
    
    if data["girisim_metni_uzunlugu"].loc[0]=="cok":
        data["sentiment"] = pd.Series(get_sentiment(data["girisimcilikle_ilgili_deneyiminizi_aciklayabilir_misiniz"].loc[0])).replace({"LABEL_1":"pozitif","LABEL_2":"negatif","LABEL_0":"nötr"})
    else:
        data["sentiment"] = np.NaN
    #Girişimcilik Metni ve STK Metninden gerekli çıkarımlar yapıldıktan sonra atıldı, burs alma durumundan gerekli çıkarımlar yapıldıktan sonra
    #modelin önem verdiği featurelar arasında 0 puan aldıkları için hesap maliyetini azaltma niyetiyle çıkarıldılar.
    #başvuru yılı, hesaplamalarızdaki işlevini tamamladı ve modele verilmeyeceği için çıkarıldı
    data = data.drop(["girisimcilikle_ilgili_deneyiminizi_aciklayabilir_misiniz","yas",
                      "hangi_stknin_uyesisiniz"]+data.columns[data.columns.str.contains("burs")].to_list(), axis=1)
    
    return data

