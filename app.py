import pickle
import streamlit as st

filename = "model.sv"
model = pickle.load(open(filename,'rb'))

def main():

    st.set_page_config(page_title="Titanic predictor")
    overview = st.container()
    # left, right = st.columns(2)
    prediction = st.container()

    st.image("https://pacjent.gov.pl/sites/default/files/styles/hero_image/public/2021-02/Walentynki%20-%20Hero_desktop%20%E2%80%93%202400x868.png?itok=qdr3KwEP")

    with overview:
        st.title("Health predictor")
        objawy_slider = st.slider("Liczba objawów", value=1, min_value=1, max_value=5)
        wiek_slider = st.slider("Wiek", value=1, min_value=1, max_value=77)
        choroby_wsp_slider = st.slider("Choroby współistniejące", value=0, min_value=0, max_value=5)
        wzrost_slider = st.slider("Wzrost", value=180, min_value=159, max_value=200)
        leki_slider = st.slider("Przyjmowane leki", value=1, min_value=1, max_value=4)



    data = [[objawy_slider, wiek_slider, choroby_wsp_slider, wzrost_slider, leki_slider]]
    zdrowie = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Przewidywana wartość zmiennej \"zdrowie\"")
        st.subheader(("1" if zdrowie[0] == 1 else "0"))
        st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][zdrowie][0] * 100))

if __name__ == "__main__":
    main()