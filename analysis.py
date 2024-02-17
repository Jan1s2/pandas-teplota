import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from enum import Enum

from functools import partial

class Seasons(Enum):
    WINTER = 1
    SPRING = 2
    SUMMER = 3
    AUTUMN = 4

class Holidays(Enum):
    NEW_YEAR = 1
    EASTER = 2
    CHRISTMAS = 3
    HALLOWEEN = 4


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

    def get_pure_data(self):
        if self.data is None:
            print("No data to analyze. Please read the data first.")
            return

        return self.data

    def get_data(self, time, base=None, filter_func=None) -> pd.DataFrame|None:
        data = self.get_pure_data()
        if data is None:
            print("No data to analyze. Please read the data first.")
            return None

        if base == 'SRA':
            data = data.dropna(subset=['SRA'])
        if filter_func is not None:
            data = data[filter_func(data)]
        if base is None:
            return data.groupby(time)
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

    def plot_monthly_avg_temperature(self, filter_func=None):
        monthly_avg_temp_data = self.get_data('měsíc', 'T-AVG', filter_func=filter_func)
        if monthly_avg_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return
        monthly_avg_temp = monthly_avg_temp_data.mean()
        self.plot("Monthly Average Temperature", "Month", "Temperature (°C)", monthly_avg_temp)

    def plot_monthly_max_temperature(self, filter_func=None):
        monthly_max_temp_data = self.get_data('měsíc', 'TMA', filter_func=filter_func)
        if monthly_max_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_max_temp = monthly_max_temp_data.max()
        self.plot("Monthly Max Temperature", "Month", "Temperature (°C)", monthly_max_temp)

    def plot_monthly_min_temperature(self, filter_func=None):
        monthly_min_temp_data = self.get_data('měsíc', 'TMI', filter_func=filter_func)
        if monthly_min_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_min_temp = monthly_min_temp_data.min()
        self.plot("Monthly Min Temperature", "Month", "Temperature (°C)", monthly_min_temp)

    def plot_monthly_all_temperature(self, filter_func=None):
        temperature = partial(self.get_data, base='T-AVG', filter_func=filter_func)
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

    def plot_yearly_all_temperature(self, filter_func=None):
        temperature = partial(self.get_data, time='rok', filter_func=filter_func)
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

    def plot_monthly_avg_rainfall(self, filter_func=None):
        monthly_avg_rainfall_data = self.get_data('měsíc', 'SRA', filter_func=filter_func)
        if monthly_avg_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_avg_rainfall = monthly_avg_rainfall_data.mean()
        self.plot("Monthly Average Rainfall", "Month", "Rainfall (mm)", monthly_avg_rainfall)

    def plot_monthly_max_rainfall(self, filter_func=None):
        monthly_max_rainfall_data = self.get_data('měsíc', 'SRA', filter_func=filter_func)
        if monthly_max_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_max_rainfall = monthly_max_rainfall_data.max()
        self.plot("Monthly Max Rainfall", "Month", "Rainfall (mm)", monthly_max_rainfall)

    def plot_monthly_min_rainfall(self, filter_func=None):
        monthly_min_rainfall_data = self.get_data('měsíc', 'SRA',  filter_func=filter_func)
        if monthly_min_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        monthly_min_rainfall = monthly_min_rainfall_data.min()
        self.plot("Monthly Min Rainfall", "Month", "Rainfall (mm)", monthly_min_rainfall)

    def plot_monthly_all_rainfall(self, filter_func=None):
        rainfall = self.get_data('měsíc', 'SRA', filter_func=filter_func)
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

    def plot_day_avg_temperature(self, filter_func=None):
        daily_avg_temp_data = self.get_data('den', 'T-AVG', filter_func=filter_func)
        if daily_avg_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        daily_avg_temp = daily_avg_temp_data.mean()
        self.plot("Daily Average Temperature", "Day", "Temperature (°C)", daily_avg_temp)

    def plot_day_of_month_max_temperature(self, filter_func=None):
        daily_max_temp_data = self.get_data('den', 'TMA', filter_func=filter_func)
        if daily_max_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        daily_max_temp = daily_max_temp_data.max()
        self.plot("Daily Max Temperature", "Day", "Temperature (°C)", daily_max_temp)

    def plot_day_of_month_min_temperature(self, filter_func=None):
        daily_min_temp_data = self.get_data('den', 'TMI', filter_func=filter_func)
        if daily_min_temp_data is None:
            print("No data to analyze. Please read the data first.")
            return

        daily_min_temp = daily_min_temp_data.min()
        self.plot("Daily Min Temperature", "Day", "Temperature (°C)", daily_min_temp)

    def plot_day_of_month_avg_rainfall(self, filter_func=None):
        daily_avg_rainfall_data = self.get_data('den', 'SRA', filter_func=filter_func)
        if daily_avg_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        daily_avg_rainfall = daily_avg_rainfall_data.mean()
        self.plot("Daily Average Rainfall", "Day", "Rainfall (mm)", daily_avg_rainfall)

    def plot_day_of_month_max_rainfall(self, filter_func=None):
        daily_max_rainfall_data = self.get_data('den', 'SRA', filter_func=filter_func)
        if daily_max_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        daily_max_rainfall = daily_max_rainfall_data.max()
        self.plot("Daily Max Rainfall", "Day", "Rainfall (mm)", daily_max_rainfall)

    def plot_day_of_month_min_rainfall(self, filter_func=None):
        daily_min_rainfall_data = self.get_data('den', 'SRA', filter_func=filter_func)
        if daily_min_rainfall_data is None:
            print("No data to analyze. Please read the data first.")
            return

        daily_min_rainfall = daily_min_rainfall_data.min()
        self.plot("Daily Min Rainfall", "Day", "Rainfall (mm)", daily_min_rainfall)

    def plot_rainfall_mean_development_decades(self, filter_func=None):
        rainfall = self.get_data('rok', 'SRA', filter_func=filter_func)
        if rainfall is None:
            print("No data to analyze. Please read the data first.")
            return

        rainfall_mean = rainfall.mean()
        rainfall_mean_decades = rainfall_mean.groupby(rainfall_mean.index // 10 * 10).mean()
        self.plot("Rainfall Mean Development Decades", "Decade", "Rainfall (mm)", rainfall_mean_decades)

    def plot_rainfall_mean_development_years(self, number_of_years=1, filter_func=None):
        rainfall = self.get_data('rok', 'SRA', filter_func=filter_func)
        if rainfall is None:
            print("No data to analyze. Please read the data first.")
            return

        rainfall_mean = rainfall.mean()
        rainfall_mean_years = rainfall_mean.groupby(rainfall_mean.index // number_of_years * number_of_years).mean()
        self.plot(f"Rainfall Mean Development Years (By {number_of_years} years)", "Year", "Rainfall (mm)", rainfall_mean_years)

    def plot_rainfall_mean_development_month(self, month):
        rainfall = self.get_data('rok', 'SRA', filter_func=lambda x: x['měsíc'] == month)
        if rainfall is None:
            print("No data to analyze. Please read the data first.")
            return

        rainfall_mean = rainfall.mean()
        self.plot(f"Rainfall Mean Development For A month", "Year", "Rainfall (mm)", rainfall_mean)

    def get_highest_rainfall_date(self, filter_func=None):
        rainfall = self.get_pure_data()
        if rainfall is None:
            print("No data to analyze. Please read the data first.")
            return
        if filter_func is not None:
            rainfall = rainfall[filter_func(rainfall)]
        index = rainfall['SRA'].idxmax()
        return (rainfall['rok'][index], rainfall['měsíc'][index], rainfall['den'][index], rainfall['SRA'][index])

    def get_lowest_rainfall_date(self, filter_func=None):
        rainfall = self.get_pure_data()
        if rainfall is None:
            print("No data to analyze. Please read the data first.")
            return
        if filter_func is not None:
            rainfall = rainfall[filter_func(rainfall)]
        index = rainfall['SRA'].idxmin()
        return (rainfall['rok'][index], rainfall['měsíc'][index], rainfall['den'][index], rainfall['SRA'][index])

    def get_highest_temperature_date(self, filter_func=None):
        temperature = self.get_pure_data()
        if temperature is None:
            print("No data to analyze. Please read the data first.")
            return
        temperature = temperature[filter_func(temperature)]
        index = temperature['T-AVG'].idxmax()
        return (temperature['rok'][index], temperature['měsíc'][index], temperature['den'][index], temperature['T-AVG'][index])

    def get_lowest_temperature_date(self, filter_func=None):
        temperature = self.get_pure_data()
        if temperature is None:
            print("No data to analyze. Please read the data first.")
            return
        temperature = temperature[filter_func(temperature)]
        index = temperature['T-AVG'].idxmin()
        return (temperature['rok'][index], temperature['měsíc'][index], temperature['den'][index], temperature['T-AVG'][index])
    
    def __get_outliers(self, column, offset=3, filter_func=None):
        data = self.get_pure_data().dropna(subset=[column])
        if data is None:
            print("No data to analyze. Please read the data first.")
            return
        if filter_func is not None:
            data = data[filter_func(data)]
        z = np.abs(stats.zscore(data[column]))
        return data[(z > offset) | (z < -offset)]

    def get_temperature_outliers(self, offset=3, filter_func=None):
        return self.__get_outliers('T-AVG', offset, filter_func)
    
    def get_rainfall_outliers(self, offset=3, filter_func=None):
        return self.__get_outliers('SRA', offset, filter_func)
    
    def get_max_temperature_outliers(self, offset=3, filter_func=None):
        return self.__get_outliers('TMA', offset, filter_func)
    
    def get_min_temperature_outliers(self, offset=3, filter_func=None):
        return self.__get_outliers('TMI', offset, filter_func)

    def filter_month(self, start_month, end_month=None):
        if end_month is None:
            return lambda x: x['měsíc'] == start_month
        if end_month < start_month:
            return lambda x: (x['měsíc'] >= start_month) | (x['měsíc'] <= end_month)
        return lambda x: (x['měsíc'] >= start_month) & (x['měsíc'] <= end_month)
    def filter_year(self, start_year, end_year=None):
        if end_year is None:
            return lambda x: x['rok'] == start_year
        return lambda x: (x['rok'] >= start_year) & (x['rok'] <= end_year)
    def filter_month_year(self, start_month, start_year, end_month=None, end_year=None):
        return lambda x: (self.filter_month(start_month, end_month)(x)) & (self.filter_year(start_year, end_year)(x))

    def filter_season(self, season:Seasons):
        match season:
            case Seasons.WINTER:
                return self.filter_month(12, 2)
            case Seasons.SPRING:
                return self.filter_month(3, 5)
            case Seasons.SUMMER:
                return self.filter_month(6, 8)
            case Seasons.AUTUMN:
                return self.filter_month(9, 11)
            case _:
                return None
    def filter_day(self, start_day, end_day=None):
        if end_day is None:
            return lambda x: x['den'] == start_day
        if end_day < start_day:
            return lambda x: (x['den'] >= start_day) | (x['den'] <= end_day)
        return lambda x: (x['den'] >= start_day) & (x['den'] <= end_day)
    def filter_date(self, start_day, start_month, end_day=None, end_month=None):
        return lambda x: (self.filter_month(start_month, end_month)(x)) & (self.filter_day(start_day, end_day)(x))
    def filter_holiday(self, holiday:Holidays):
        match holiday:
            case Holidays.NEW_YEAR:
                return self.filter_date(1, 1)
            case Holidays.EASTER:
                return self.filter_date(1, 4, 30, 4)
            case Holidays.CHRISTMAS:
                return self.filter_date(24, 12, 26, 12)
            case Holidays.HALLOWEEN:
                return self.filter_date(31, 10)
            case _:
                return None
    def filter_decade(self, start_year, end_year=None):
        if end_year is None:
            return lambda x: (x['rok'] // 10) * 10 == start_year
        return lambda x: (x['rok'] // 10) * 10 >= start_year & (x['rok'] // 10) * 10 <= end_year

    # returns the hottest day of year for each year
    def hottest_day_of_year(self, filter_func=None):
        data = self.get_pure_data()
        if data is None:
            print("No data to analyze. Please read the data first.")
            return
        if filter_func is not None:
            data = data[filter_func(data)]
        idx = data.groupby('rok')['TMA'].idxmax()
        return data.loc[idx]

    def coldest_day_of_year(self, filter_func=None):
        data = self.get_pure_data()
        if data is None:
            print("No data to analyze. Please read the data first.")
            return
        if filter_func is not None:
            data = data[filter_func(data)]
        idx = data.groupby('rok')['TMI'].idxmin()
        return data.loc[idx]

    def rainiest_day_of_year(self, filter_func=None):
        data = self.get_pure_data()
        if data is None:
            print("No data to analyze. Please read the data first.")
            return
        if filter_func is not None:
            data = data[filter_func(data)]
        idx = data.groupby('rok')['SRA'].idxmax()
        return data.loc[idx]

    def get_year_max_temperature(self, year):
        return self.hottest_day_of_year(self.filter_year(year))
    def get_year_min_temperature(self, year):
        return self.coldest_day_of_year(self.filter_year(year))

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
    # analyzer.plot_rainfall_mean_development_decades()
    # analyzer.plot_rainfall_mean_development_years(10)
    # analyzer.plot_rainfall_mean_development_years(1, 1900, 2020)
    # analyzer.plot_rainfall_mean_development_month(1)
    # analyzer.plot_day_of_month_max_rainfall()
    # print(analyzer.get_highest_rainfall_date())
    # print(analyzer.get_lowest_rainfall_date())
    # print(analyzer.get_highest_temperature_date(1999, 1999))
    # analyzer.plot_day_of_month_avg_rainfall()
    # print(analyzer.get_rainfall_outliers())
    # print(analyzer.get_max_temperature_outliers())
    # print(analyzer.get_min_temperature_outliers())
    # print(analyzer.get_temperature_outliers(filter_func=analyzer.filter_month(1)))
    print(analyzer.get_highest_temperature_date(analyzer.filter_holiday(Holidays.CHRISTMAS)))
    print(analyzer.hottest_day_of_year(analyzer.filter_year(1999)))
    print(analyzer.get_temperature_outliers(offset=4))
    analyzer.plot_monthly_avg_temperature(filter_func=analyzer.filter_holiday(Holidays(1)))

