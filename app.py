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
def vislog(inpefa, x, filename):
    plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
    plt.figtext(0.1, 0.95, filename, fontsize=12, ha='left')  # Adding the file name at the top left
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
    # Create a temporary file to store the LAS data
    with tempfile.NamedTemporaryFile(delete=False, suffix='.las') as tmpfile:
        tmpfile.write(file.getvalue())
        tmpfile.seek(0)
        las = lasio.read(tmpfile.name).df()
        y = las.GR.dropna()
        x = np.array(y.index)
        return inpefa(y, x), x, file.name  # Using the original file name

def main():
    st.title('Well Log Analysis')
    uploaded_files = st.file_uploader("Choose LAS files", type=['.las'], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            inpefa_log, x, filename = process_file(uploaded_file)
            vislog(inpefa_log, x, filename)
            # Create a DataFrame to save to CSV
            df = pd.DataFrame(inpefa_log)
            df['Depth'] = x
            csv = df.to_csv(index=False)
            # Remove the '.las' and replace with '.csv' for the download filename
            download_filename = filename.replace('.las', '.csv')
            st.download_button(label="Download CSV", data=csv, file_name=download_filename, mime='text/csv')
    else:
        st.write('Please upload LAS files.')

if __name__ == "__main__":
    main()
