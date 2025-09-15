
import pandas as pd

# Disabled generation of synthetic data because it was too resource-intensive for POC demos and testing
def generate_mock(data: pd.DataFrame) -> pd.DataFrame:
    return data
    discrete_columns = data.columns.tolist()

    ctgan = CTGAN(epochs=10)
    ctgan.fit(data, discrete_columns)

    synthetic_data = ctgan.sample(500)

    return synthetic_data