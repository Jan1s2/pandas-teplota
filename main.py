import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from enum import Enum


class Types(Enum):
    MEAN = 1
    MAX = 2
    MIN = 3

class TemperatureDataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        xls = pd.ExcelFile(self.file_path)
        self.data = pd.read_excel(xls, 'data')

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

    def get_data(self, time, base, variant):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return
        
        if base == 'SRA':
            data = self.data.dropna(subset=['SRA'])
        else:
            data = self.data
        match variant:
            case Types.MEAN:
                return data.groupby(time)[base].mean()
            case Types.MAX:
                return data.groupby(time)[base].max()
            case Types.MIN:
                return data.groupby(time)[base].min()
            case _:
                return None

    def monthly_avg_temperature(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_avg_temp = self.data.groupby('měsíc')['T-AVG'].mean()
        return monthly_avg_temp

    def monthly_max_temperature(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_max_temp = self.data.groupby('měsíc')['TMA'].max()
        return monthly_max_temp

    def monthly_min_temperature(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_min_temp = self.data.groupby('měsíc')['TMI'].min()
        return monthly_min_temp

    def yearly_max_temperature(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        yearly_max_temp = self.data.groupby('rok')['TMA'].max()
        return yearly_max_temp
    def yearly_min_temperature(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        yearly_min_temp = self.data.groupby('rok')['TMI'].min()
        return yearly_min_temp
    def yearly_avg_temperature(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        yearly_avg_temp = self.data.groupby('rok')['T-AVG'].mean()
        return yearly_avg_temp

    def monthly_avg_rainfall(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_avg_rainfall = self.data.dropna(subset=['SRA']).groupby('měsíc')['SRA'].mean()
        return monthly_avg_rainfall
    def monthly_max_rainfall(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_max_rainfall = self.data.dropna(subset=['SRA']).groupby('měsíc')['SRA'].max()
        return monthly_max_rainfall
    def monthly_min_rainfall(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_min_rainfall = self.data.dropna(subset=['SRA']).groupby('měsíc')['SRA'].min()
        return monthly_min_rainfall

    def yearly_max_rainfall(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        yearly_max_rainfall = self.data.dropna(subset=['SRA']).groupby('rok')['SRA'].max()
        return yearly_max_rainfall
    
    def yearly_min_rainfall(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        yearly_min_rainfall = self.data.dropna(subset=['SRA']).groupby('rok')['SRA'].min()
        return yearly_min_rainfall

    def yearly_avg_rainfall(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        yearly_avg_rainfall = self.data.dropna(subset=['SRA']).groupby('rok')['SRA'].mean()
        return yearly_avg_rainfall

    def plot(self, title, xlabel, ylabel, *data):
        if data is None:
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
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_avg_temp = self.monthly_avg_temperature()
        self.plot("Monthly Average Temperature", "Month", "Temperature (°C)", monthly_avg_temp)

    def plot_monthly_max_temperature(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_max_temp = self.monthly_max_temperature()
        self.plot("Monthly Max Temperature", "Month", "Temperature (°C)", monthly_max_temp)

    def plot_monthly_min_temperature(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_min_temp = self.monthly_min_temperature()
        self.plot("Monthly Min Temperature", "Month", "Temperature (°C)", monthly_min_temp)

    def plot_monthly_all_temperature(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_avg_temp = self.monthly_avg_temperature()
        monthly_max_temp = self.monthly_max_temperature()
        monthly_min_temp = self.monthly_min_temperature()
        self.plot("Monthly Temperature", "Month", "Temperature (°C)", monthly_avg_temp, monthly_max_temp, monthly_min_temp)

    def plot_monthly_rainfall(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_avg_rainfall = self.monthly_avg_rainfall()
        self.plot("Monthly Average Rainfall", "Month", "Rainfall (mm)", monthly_avg_rainfall)

    def plot_monthly_max_rainfall(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_max_rainfall = self.monthly_max_rainfall()
        self.plot("Monthly Max Rainfall", "Month", "Rainfall (mm)", monthly_max_rainfall)
    def plot_monthly_min_rainfall(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_min_rainfall = self.monthly_min_rainfall()
        self.plot("Monthly Min Rainfall", "Month", "Rainfall (mm)", monthly_min_rainfall)

    def plot_monthly_all_rainfall(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_avg_rainfall = self.monthly_avg_rainfall()
        monthly_avg_rainfall.name = "Average Rainfall"
        monthly_max_rainfall = self.monthly_max_rainfall()
        monthly_max_rainfall.name = "Max Rainfall"
        monthly_min_rainfall = self.monthly_min_rainfall()
        monthly_min_rainfall.name = "Min Rainfall"
        self.plot("Monthly Rainfall", "Month", "Rainfall (mm)", monthly_avg_rainfall, monthly_max_rainfall, monthly_min_rainfall)

if __name__ == "__main__":
    file_path = "klementinum.xlsx"  # Update with your file path
    analyzer = TemperatureDataAnalyzer(file_path)
    analyzer.read_data()
    analyzer.summary_statistics()
    print(analyzer.monthly_avg_temperature())
    print(analyzer.yearly_max_rainfall())
    print(analyzer.yearly_avg_rainfall())
    print(analyzer.yearly_min_rainfall())
    print(analyzer.monthly_max_rainfall())
    print(analyzer.monthly_avg_rainfall())
    print(analyzer.monthly_min_rainfall())
    analyzer.plot_monthly_avg_temperature()
    analyzer.plot_monthly_all_rainfall()
