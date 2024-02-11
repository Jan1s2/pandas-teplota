import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from enum import Enum

from functools import partial


class Types(Enum):
    MEAN = 1
    MAX = 2
    MIN = 3

class TemperatureDataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        xls = pd.ExcelFile(self.file_path)
        # rok: Year
        # měsíc: Month
        # den: Day
        # T-AVG: Average temperature (°C)
        # TMA: Maximum temperature (°C)
        # TMI: Minimum temperature (°C)
        # SRA: Total rainfall (mm)
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

    def get_data(self, time, base, variant, filter_func=None):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return
        
        if base == 'SRA':
            data = self.data.dropna(subset=['SRA'])
        else:
            data = self.data
        if filter_func is not None:
            data = data[filter_func(data)]
        match variant:
            case Types.MEAN:
                return data.groupby(time)[base].mean()
            case Types.MAX:
                return data.groupby(time)[base].max()
            case Types.MIN:
                return data.groupby(time)[base].min()
            case _:
                return None


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

        monthly_avg_temp = self.get_data('měsíc', 'T-AVG', Types.MEAN)
        self.plot("Monthly Average Temperature", "Month", "Temperature (°C)", monthly_avg_temp)

    def plot_monthly_max_temperature(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_max_temp = self.get_data('měsíc', 'TMA', Types.MAX)
        self.plot("Monthly Max Temperature", "Month", "Temperature (°C)", monthly_max_temp)

    def plot_monthly_min_temperature(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_min_temp = self.get_data('měsíc', 'TMI', Types.MIN)
        self.plot("Monthly Min Temperature", "Month", "Temperature (°C)", monthly_min_temp)

    def plot_monthly_all_temperature(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return
        temperature = partial(self.get_data, 'měsíc')

        monthly_avg_temp = temperature('T-AVG', Types.MEAN)
        monthly_avg_temp.name = "Average Temperature"
        monthly_max_temp = temperature('TMA', Types.MAX)
        monthly_max_temp.name = "Max Temperature"
        monthly_min_temp = temperature('TMI', Types.MIN)
        monthly_min_temp.name = "Min Temperature"
        self.plot("Monthly Temperature", "Month", "Temperature (°C)", monthly_avg_temp, monthly_max_temp, monthly_min_temp)

    def plot_yearly_all_temperature(self, start_year=0, end_year=10000):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return
        # start year 1900
        # end year 2020
        temperature = partial(self.get_data, 'rok', filter_func=lambda x: (x['rok'] >= start_year) & (x['rok'] <= end_year))
        # temperature = partial(self.get_data, 'rok')

        monthly_avg_temp = temperature('T-AVG', Types.MEAN)
        monthly_avg_temp.name = "Average Temperature"
        monthly_max_temp = temperature('TMA', Types.MAX)
        monthly_max_temp.name = "Max Temperature"
        monthly_min_temp = temperature('TMI', Types.MIN)
        monthly_min_temp.name = "Min Temperature"
        self.plot("Yearly Temperature", "Year", "Temperature (°C)", monthly_avg_temp, monthly_max_temp, monthly_min_temp)

    def plot_monthly_avg_rainfall(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_avg_rainfall = self.get_data('měsíc', 'SRA', Types.MEAN)
        self.plot("Monthly Average Rainfall", "Month", "Rainfall (mm)", monthly_avg_rainfall)

    def plot_monthly_max_rainfall(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_max_rainfall = self.get_data('měsíc', 'SRA', Types.MAX)
        self.plot("Monthly Max Rainfall", "Month", "Rainfall (mm)", monthly_max_rainfall)

    def plot_monthly_min_rainfall(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_min_rainfall = self.get_data('měsíc', 'SRA', Types.MIN)
        self.plot("Monthly Min Rainfall", "Month", "Rainfall (mm)", monthly_min_rainfall)

    def plot_monthly_all_rainfall(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return
        rainfall = partial(self.get_data, 'měsíc', 'SRA')

        monthly_avg_rainfall = rainfall(Types.MEAN)
        monthly_avg_rainfall.name = "Average Rainfall"
        monthly_max_rainfall = rainfall(Types.MAX)
        monthly_max_rainfall.name = "Max Rainfall"
        monthly_min_rainfall = rainfall(Types.MIN)
        monthly_min_rainfall.name = "Min Rainfall"
        self.plot("Monthly Rainfall", "Month", "Rainfall (mm)", monthly_avg_rainfall, monthly_max_rainfall, monthly_min_rainfall)

if __name__ == "__main__":
    file_path = "klementinum.xlsx"  # Update with your file path
    analyzer = TemperatureDataAnalyzer(file_path)
    analyzer.read_data()
    analyzer.summary_statistics()
    print(analyzer.get_data('rok', 'T-AVG', Types.MEAN))
    print(analyzer.get_data('rok', 'TMA', Types.MAX))
    print(analyzer.get_data('rok', 'TMI', Types.MIN))
    print(analyzer.get_data('rok', 'SRA', Types.MEAN))
    analyzer.plot_monthly_avg_temperature()
    analyzer.plot_monthly_all_rainfall()
    analyzer.plot_yearly_all_temperature() 
