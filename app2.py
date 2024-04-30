import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lasio
import tempfile
from PyNPEFAmain import inpefa

# Streamlit page configuration
st.set_page_config(page_title='INPEFA Well Log Analysis')

# Make header
st.title('Integrated Prediction Error Filter Analysis (INPEFA)')
st.write('This app is a Streamlit implementation of the INPEFA method for well log data.')

st.set_option('deprecation.showPyplotGlobalUse', False)

# Visualization function
def vislog(inpefa, x):
    plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
    titles = ['Original GR Curve', 'Long Term INPEFA', 'Mid Term INPEFA', 'Short Term INPEFA', 'Shorter Term INPEFA']
    for i, key in enumerate(['OG', '1', '2', '3', '4']):
        plt.subplot(1, 5, i+1)
        plt.plot(inpefa[key], -x)
        plt.grid(True)
        plt.xlabel('GR (API)' if i == 0 else '')
        plt.ylabel('Depth (ft)' if i == 0 else '')
        plt.title(titles[i])
        plt.xlim((0, 150) if key == 'OG' else (-1, 1))

    st.pyplot()

def process_file(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.las') as tmpfile:
        tmpfile.write(file.getvalue())
        tmpfile.seek(0)
        las = lasio.read(tmpfile.name).df()
        y = las.GR.dropna()
        x = np.array(y.index)
        return inpefa(y, x), x

def main():
    st.title('Well Log Analysis')
    uploaded_files = st.file_uploader("Choose LAS files", type=['.las'], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            inpefa_log, x = process_file(uploaded_file)
            vislog(inpefa_log, x)
    else:
        st.write('Please upload LAS files.')

if __name__ == "__main__":
    main()
