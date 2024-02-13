import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from functools import partial

class TemperatureDataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data:pd.DataFrame = None

    def read_data(self):
        try:
            xls = pd.ExcelFile(self.file_path)
            self.data = pd.read_excel(xls, 'data')
        except FileNotFoundError:
            print("File not found. Please provide a valid file path.")

    def summary_statistics(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        summary = self.data.describe()
        print(summary)

    def get_data(self, time, base, filter_func=None) -> pd.DataFrame|None:
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return None

        data = self.data
        if base == 'SRA':
            data = data.dropna(subset=['SRA'])
        if filter_func is not None:
            data = data[filter_func(data)]

        return data.groupby(time)[base]

    def plot(self, title, xlabel, ylabel, *data):
        if not data:
            print("No data to plot. Please provide valid data.")
            return
        for i in data:
            i.plot(label=i.name)
        plt.title(title)
        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
    
    def plot_monthly_avg_temperature(self):
        monthly_avg_temp_data = self.get_data('měsíc', 'T-AVG')
        if monthly_avg_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_avg_temp = monthly_avg_temp_data.mean()
        self.plot("Monthly Average Temperature", "Month", "Temperature (°C)", monthly_avg_temp)

    def plot_monthly_max_temperature(self):
        monthly_max_temp_data = self.get_data('měsíc', 'TMA')
        if monthly_max_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_max_temp = monthly_max_temp_data.max()
        self.plot("Monthly Max Temperature", "Month", "Temperature (°C)", monthly_max_temp)

    def plot_monthly_min_temperature(self):
        monthly_min_temp_data = self.get_data('měsíc', 'TMI')
        if monthly_min_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_min_temp = monthly_min_temp_data.min()
        self.plot("Monthly Min Temperature", "Month", "Temperature (°C)", monthly_min_temp)

    def plot_monthly_all_temperature(self):
        temperature = partial(self.get_data, base='T-AVG')
        monthly_avg_temp_data = temperature('měsíc')
        if monthly_avg_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_avg_temp = monthly_avg_temp_data.mean()
        monthly_avg_temp.name = "Average Temperature"
        monthly_max_temp = temperature('TMA').max()
        monthly_max_temp.name = "Max Temperature"
        monthly_min_temp = temperature('TMI').min()
        monthly_min_temp.name = "Min Temperature"
        self.plot("Monthly Temperature", "Month", "Temperature (°C)", monthly_avg_temp, monthly_max_temp, monthly_min_temp)

    def plot_yearly_all_temperature(self, start_year=0, end_year=10000):
        temperature = partial(self.get_data, time='rok', filter_func=lambda x: (x['rok'] >= start_year) & (x['rok'] <= end_year))
        monthly_avg_temp_data = temperature(base='T-AVG')
        if monthly_avg_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_avg_temp:pd.DataFrame = monthly_avg_temp_data.mean()
        monthly_avg_temp.name = "Average Temperature"
        monthly_max_temp = temperature(base='TMA').max()
        monthly_max_temp.name = "Max Temperature"
        monthly_min_temp = temperature(base='TMI').min()
        monthly_min_temp.name = "Min Temperature"
        self.plot("Yearly Temperature", "Year", "Temperature (°C)", monthly_avg_temp, monthly_max_temp, monthly_min_temp)

    def plot_monthly_avg_rainfall(self):
        monthly_avg_rainfall_data = self.get_data('měsíc', 'SRA')
        if monthly_avg_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_avg_rainfall = monthly_avg_rainfall_data.mean()
        self.plot("Monthly Average Rainfall", "Month", "Rainfall (mm)", monthly_avg_rainfall)

    def plot_monthly_max_rainfall(self):
        monthly_max_rainfall_data = self.get_data('měsíc', 'SRA')
        if monthly_max_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_max_rainfall = monthly_max_rainfall_data.max()
        self.plot("Monthly Max Rainfall", "Month", "Rainfall (mm)", monthly_max_rainfall)

    def plot_monthly_min_rainfall(self):
        monthly_min_rainfall_data = self.get_data('měsíc', 'SRA')
        if monthly_min_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_min_rainfall = monthly_min_rainfall_data.min()
        self.plot("Monthly Min Rainfall", "Month", "Rainfall (mm)", monthly_min_rainfall)

    def plot_monthly_all_rainfall(self):
        rainfall = self.get_data('měsíc', 'SRA')
        if rainfall is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_avg_rainfall = rainfall.mean()
        monthly_avg_rainfall.name = "Average Rainfall"
        monthly_max_rainfall = rainfall.max()
        monthly_max_rainfall.name = "Max Rainfall"
        monthly_min_rainfall = rainfall.min()
        monthly_min_rainfall.name = "Min Rainfall"
        self.plot("Monthly Rainfall", "Month", "Rainfall (mm)", monthly_avg_rainfall, monthly_max_rainfall, monthly_min_rainfall)

    def plot_day_of_month_avg_temperature(self):
        daily_avg_temp_data = self.get_data('den', 'T-AVG')
        if daily_avg_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        daily_avg_temp = daily_avg_temp_data.mean()
        self.plot("Daily Average Temperature", "Day", "Temperature (°C)", daily_avg_temp)

    def plot_day_of_month_max_temperature(self):
        daily_max_temp_data = self.get_data('den', 'TMA')
        if daily_max_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        daily_max_temp = daily_max_temp_data.max()
        self.plot("Daily Max Temperature", "Day", "Temperature (°C)", daily_max_temp)

    def plot_day_of_month_min_temperature(self):
        daily_min_temp_data = self.get_data('den', 'TMI')
        if daily_min_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        daily_min_temp = daily_min_temp_data.min()
        self.plot("Daily Min Temperature", "Day", "Temperature (°C)", daily_min_temp)

    def plot_day_of_month_avg_rainfall(self):
        daily_avg_rainfall_data = self.get_data('den', 'SRA')
        if daily_avg_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        daily_avg_rainfall = daily_avg_rainfall_data.mean()
        self.plot("Daily Average Rainfall", "Day", "Rainfall (mm)", daily_avg_rainfall)

    def plot_day_of_month_max_rainfall(self):
        daily_max_rainfall_data = self.get_data('den', 'SRA')
        if daily_max_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        daily_max_rainfall = daily_max_rainfall_data.max()
        self.plot("Daily Max Rainfall", "Day", "Rainfall (mm)", daily_max_rainfall)

    def plot_day_of_month_min_rainfall(self):
        daily_min_rainfall_data = self.get_data('den', 'SRA')
        if daily_min_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        daily_min_rainfall = daily_min_rainfall_data.min()
        self.plot("Daily Min Rainfall", "Day", "Rainfall (mm)", daily_min_rainfall)

    def diff_rain(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return
        data = self.get_dat
        print(data)

    def plot_diff_rain(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return
        # data = self.get_data('rok', 'SRA', filter_func=lambda x: (x['rok'] == 2019) | (x['rok'] == 2020)).diff()
        def filter_func(x):
            return (x['rok'] == 2019) | (x['rok'] == 2020)
        data = self.data.dropna(subset=['SRA'])
        data = data[filter_func(data)]
        data = data['SRA'].diff()
        data.name = "INFO"
        self.plot("Daily Rainfall Difference", "Day", "Rainfall (mm)", data)


if __name__ == "__main__":
    file_path = "klementinum.xlsx"  # Update with your file path
    analyzer = TemperatureDataAnalyzer(file_path)
    analyzer.read_data()
    # analyzer.summary_statistics()
    # print(analyzer.get_data('rok', 'T-AVG').mean())
    # print(analyzer.get_data('rok', 'TMA').max())
    # print(analyzer.get_data('rok', 'TMI').min())
    # print(analyzer.get_data('rok', 'SRA').mean())
    # analyzer.plot_monthly_avg_temperature()
    # analyzer.plot_monthly_all_rainfall()
    # analyzer.plot_yearly_all_temperature(1900, 2020)
    analyzer.plot_diff_rain()
