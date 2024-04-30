import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyNPEFAmain import inpefa
import lasio
import tempfile

# Streamlit page configuration
st.set_page_config(page_title='INPEFA Well Log Analysis')

# Make header
st.title('Integrated Prediction Error Filter Analysis (INPEFA)')
st.write('This app is a Streamlit implementation of the INPEFA method for well log data.')

st.set_option('deprecation.showPyplotGlobalUse', False)
# Visualization function
def vislog(inpefa, x):
    if inpefa is not None:
        plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')

        # Original signal
        plt.subplot(151)
        plt.plot(inpefa['OG'], -x)
        plt.grid(True)
        plt.xlabel('GR (API)')
        plt.ylabel('Depth (ft)')
        plt.xlim((0, 150))
        plt.title('Original GR Curve')

        # Long term INPEFA
        plt.subplot(152)
        plt.plot(inpefa['1'], -x)
        plt.grid(True)
        plt.xlim((-1, 1))
        plt.title('Long Term INPEFA')

        # Mid term INPEFA
        plt.subplot(153)
        plt.plot(inpefa['2'], -x)
        plt.grid(True)
        plt.xlim((-1, 1))
        plt.title('Mid Term INPEFA')

        # Short term INPEFA
        plt.subplot(154)
        plt.plot(inpefa['3'], -x)
        plt.grid(True)
        plt.xlim((-1, 1))
        plt.title('Short Term INPEFA')

        # Shorter term INPEFA
        plt.subplot(155)
        plt.plot(inpefa['4'], -x)
        plt.grid(True)
        plt.xlim((-1, 1))
        plt.title('Shorter Term INPEFA')

        st.pyplot()

def main():
    st.title('Well Log Analysis')
    uploaded_file = st.file_uploader("Choose a LAS file", type=['.las'])

    if uploaded_file is not None:
        # Convert Streamlit's UploadedFile to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.las') as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            tmpfile.seek(0)
            las = lasio.read(tmpfile.name).df()  # Use the temporary file name
            y = las.GR.dropna()
            x = np.array(y.index)
            inpefa_log = inpefa(y, x)
            vislog(inpefa_log, x)
    else:
        st.write('Please upload a LAS file.')

if __name__ == "__main__":
    main()







