import streamlit as st
from catboost import CatBoostRegressor
import base64
import funcs
import pandas as pd
import numpy as np
st.set_page_config(layout="centered",page_title="Girişimcilik Değerlendirme Formu", page_icon=":dart:")


@st.cache_data
def read_columns():
    return pd.read_csv("Streamlit/data/unique_columns.csv")

@st.cache_resource
def load_cat_models():
    loaded_models = []
    for i in range(20):
        model_path = f"catboost_model_{i}.cbm"
        model = CatBoostRegressor()
        model.load_model("Streamlit/models/" + model_path)
        loaded_models.append(model)
    return loaded_models

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

background = get_base64("Streamlit/media/background.jpg")

with open("Streamlit/style/style.css", "r") as style:
    css=f"""<style>{style.read().format(background=background)}</style>"""
    st.markdown(css, unsafe_allow_html=True)



uniqueler = read_columns()

kisisel, egitim, ekstra = st.tabs(["🧑 Kisisel","🎓 Eğitim","🍀 Diğer"])
col1, col2 = kisisel.columns(2)
col3, col4 = egitim.columns(2)
col5, col6 = ekstra.columns(2)

def veri_giris_formu():
    # Kullanıcıdan temel bilgiler ve Aile bilgileri
    with col1:
        cinsiyet = st.selectbox("Cinsiyet", ["Erkek", "Kadın", "Belirtmek istemiyorum"])
        yas = st.number_input("Yaşınız", min_value=17)
        dogum_yeri = st.selectbox("Doğum Yeri",sorted([x for x in uniqueler.sehirler if isinstance(x, str)]))
        ikametgah_sehri = st.selectbox("İkametgah Şehri",sorted([x for x in uniqueler.sehirler if isinstance(x, str)]))
        lise_sehir = st.selectbox("Lise Şehri",sorted([x for x in uniqueler.sehirler if isinstance(x, str)]))
        kardes_sayisi = st.number_input("Kardeş Sayısı", min_value=0, step=1)

    with col2:
        anne_egitim_durumu = st.selectbox("Anne Eğitim Durumu",
                                          ["İlkokul", "Ortaokul", "Lise", "Üniversite", "Yüksek Lisans", "Doktora"])
        anne_calisma_durumu = st.selectbox("Anne Çalışma Durumu", ["Evet", "Hayır"])
        anne_sektor = st.selectbox("Anne Sektör", ['Özel Sektör', 'Kamu', 'Diğer'],
                                   disabled={"Evet": False, "Hayır": True}.get(anne_calisma_durumu))
        if {"Evet": False, "Hayır": True}.get(anne_calisma_durumu):
            anne_sektor = np.NaN
        baba_egitim_durumu = st.selectbox("Baba Eğitim Durumu",
                                          ["İlkokul", "Ortaokul", "Lise", "Üniversite", "Yüksek Lisans", "Doktora"])
        baba_calisma_durumu = st.selectbox("Baba Çalışma Durumu", ["Evet", "Hayır"])
        baba_sektor = st.selectbox("Baba Sektör", ['Özel Sektör', 'Kamu', 'Diğer'],
                                   disabled={"Evet": False, "Hayır": True}.get(baba_calisma_durumu))
        if {"Evet": False, "Hayır": True}.get(baba_calisma_durumu):
            baba_sektor = np.NaN

    # Eğitim bilgileri

    with col3:
        universite_turu = st.selectbox("Üniversite Türü", ["Devlet", "Özel"])
        universite_sinifi = st.selectbox("Üniversite Kaçıncı Sınıf", ["Hazırlık",1,2,3,4,5,6])


        lise_turu = st.selectbox("Lise Türü", ["Devlet", "Özel"])
        lise_bolumu = st.selectbox("Lise Bölümü", ['Esit agirlik', 'Sözel', 'Sayısal', 'Dil'])

    with col4:

        universite_adi = st.selectbox("Üniversite Adı",sorted([x for x in uniqueler.universite_adlari if isinstance(x, str)]), index=None, placeholder="Bir Üniversite Adı Yazınız", help="Listede yoksa boş bırakabilirsiniz")

        bolum = st.selectbox("Üniversite Bölümü",sorted([x for x in uniqueler.bölüm if isinstance(x, str)]), index=None, placeholder="Üniversite Bölümünüzü Yazınız", help="Listede yoksa boş bırakabilirsiniz")
        lise_adi = st.text_input("Lise Adı")
        lise_mezuniyet_notu = st.number_input("Lise Mezuniyet Notu", min_value=0.0, max_value=100.0)

    with egitim:
        universite_notu = st.number_input("Üniversite Not Ortalaması", min_value=0.0, max_value=4.0, disabled = {"Hazırlık":True}.get(universite_sinifi,False))
        if {"Hazırlık":True}.get(universite_sinifi,False):
            universite_notu = 0.0
    with col5:
        # Burs bilgileri
        burs_aliyor_mu = st.selectbox("Burs Alıyor mu?", ["Evet", "Hayır"])
        profesyonel_spor_dali = st.selectbox("Profesyonel Bir Spor Dalı İle Meşgul müsünüz?", ["Evet", "Hayır"])
        girisimcilik_kulube_uye_mi = st.selectbox("Girişimcilik Kulübüne Üye misiniz?", ["Evet", "Hayır"])
        girisimcilik_deneyimi = st.selectbox("Girişimcilikle İlgili Deneyiminiz Var mı?", ["Evet", "Hayır"])
        aktif_stk_uyesi_mi = st.selectbox("Aktif Olarak Bir STK Üyesi misiniz?", ["Evet", "Hayır"])
        ingilizce_biliyor_mu = st.selectbox("İngilizce Biliyor musunuz?", ["Evet", "Hayır"])
    # Diğer bilgiler
    with col6:
        burs_aldigi_baska_kurum = st.text_input("Burs Aldığı Kurum",disabled = {"Evet":False,"Hayır":True}.get(burs_aliyor_mu))
        if {"Evet":False,"Hayır":True}.get(burs_aliyor_mu):
            burs_aldigi_baska_kurum = "bos"
        spor_dalindaki_rol = st.selectbox("Spor Dalındaki Rolünüz Nedir?",['Takım Oyuncusu', 'Bireysel Spor', 'Lider/Kaptan', 'Diğer'],
                                          disabled = {"Evet":False,"Hayır":True}.get(profesyonel_spor_dali))
        if {"Evet":False,"Hayır":True}.get(profesyonel_spor_dali):
            profesyonel_spor_dali = np.NaN
        girisimcilik_deneyimi_aciklama = st.text_area("Girişimcilik Deneyiminizi Açıklayabilir misiniz?",
                                                      disabled = {"Evet":False,"Hayır":True}.get(girisimcilik_deneyimi))
        if {"Evet":False,"Hayır":True}.get(girisimcilik_deneyimi):
            girisimcilik_deneyimi_aciklama = ""
        hangi_stk_uyesi = st.text_input("Hangi STK'nın Üyesisiniz?",disabled = {"Evet":False,"Hayır":True}.get(aktif_stk_uyesi_mi))
        if {"Evet":False,"Hayır":True}.get(aktif_stk_uyesi_mi):
            hangi_stk_uyesi = "bos"



    # Form verilerini bir dict'te toplayabiliriz:
    form_verileri = {
        "cinsiyet": cinsiyet,
        "yas": yas,
        "dogum_yeri": dogum_yeri,
        "ikametgah_sehri": ikametgah_sehri,
        "universite_adi": universite_adi,
        "universite_turu": universite_turu,
        "burs_aliyor_mu": burs_aliyor_mu,
        "bölüm": bolum,
        "universite_kacinci_sinif": universite_sinifi,
        "universite_not_ortalamasi": universite_notu,
        "lise_adi": lise_adi,
        "lise_sehir": lise_sehir,
        "lise_turu": lise_turu,
        "lise_bolumu": lise_bolumu,
        "lise_mezuniyet_notu": lise_mezuniyet_notu,
        "burs_aldigi_baska_kurum": burs_aldigi_baska_kurum,
        "anne_egitim_durumu": anne_egitim_durumu,
        "anne_calisma_durumu": anne_calisma_durumu,
        "anne_sektor": anne_sektor,
        "baba_egitim_durumu": baba_egitim_durumu,
        "baba_calisma_durumu": baba_calisma_durumu,
        "baba_sektor": baba_sektor,
        "kardes_sayisi": kardes_sayisi,
        "girisimcilik_kulupleri_tarzi_bir_kulube_uye_misiniz": girisimcilik_kulube_uye_mi,
        "profesyonel_bir_spor_daliyla_mesgul_musunuz": profesyonel_spor_dali,
        "spor_dalindaki_rolunuz_nedir": spor_dalindaki_rol,
        "aktif_olarak_bir_stk_üyesi_misiniz": aktif_stk_uyesi_mi,
        "hangi_stknin_uyesisiniz": hangi_stk_uyesi,
        "girisimcilikle_ilgili_deneyiminiz_var_mi": girisimcilik_deneyimi,
        "girisimcilikle_ilgili_deneyiminizi_aciklayabilir_misiniz": girisimcilik_deneyimi_aciklama,
        "ingilizce_biliyor_musunuz": ingilizce_biliyor_mu
    }

    return pd.DataFrame(form_verileri, index=[0])

data = veri_giris_formu()
models = load_cat_models()

cat_features = ['cinsiyet', 'dogum_yeri', 'ikametgah_sehri', 'universite_adi', 'universite_turu', 'bölüm',
 'lise_adi', 'lise_sehir', 'lise_turu', 'lise_bolumu', 'anne_egitim_durumu', 'anne_calisma_durumu', 'anne_sektor',
 'baba_egitim_durumu', 'baba_calisma_durumu', 'baba_sektor', 'girisimcilik_kulupleri_tarzi_bir_kulube_uye_misiniz',
 'profesyonel_bir_spor_daliyla_mesgul_musunuz', 'spor_dalindaki_rolunuz_nedir', 'aktif_olarak_bir_stk_üyesi_misiniz',
 'girisimcilikle_ilgili_deneyiminiz_var_mi', 'ingilizce_biliyor_musunuz', 'stk_metni_uzunlugu', 'girisim_metni_uzunlugu',
 'bu_rs_kurumu', 'lise_kategorisi', 'bölüm_kategorisi', 'sentiment']
def calculate_puan(data, models):
    new_data = funcs.feature_engineering_func(data)
    new_data[cat_features] = new_data[cat_features].fillna("Unknown")
    prediction = np.sum([model.predict(new_data) for model in models], axis=0)  / len(models)
    return new_data, prediction



if ekstra.button("Hesapla", type="primary", use_container_width=True):

    puan = str(round(calculate_puan(data,models)[1][0]))
    error_rate = 5.7
    st.markdown(
        f"""
        <div style="text-align: center; padding: 10px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #f9f9f9;">
            <h2 style="font-size: 20px; color: #4CAF50;">🎉 Girişimcilik Değerlendirme Puanınız 🎉</h2>
            <h1 style="color: #FF5722;">{puan}<span style="font-size: 15px; color: #555;">± {error_rate}</span></h1>
            <p style="font-size: 18px; color: #555;">Genel Ortalama <b>32</b></b></p>
            <p style="font-size: 18px; color: #555;">Başarılarınız için tebrikler!</p>
        </div>
        """,
        unsafe_allow_html=True
    )


