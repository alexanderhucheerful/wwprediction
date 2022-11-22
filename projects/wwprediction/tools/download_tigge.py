import calendar
import os

import numpy as np
from ecmwfapi import ECMWFService
from ecmwfapi.api import APIException
from fire import Fire

config = {
    "url": "https://apps.ecmwf.int/v1",
    "key": "ff63ab94380ba107b50528cc2bc4be5c",
    "email": "damo_meteorology_team@list.alibaba-inc.com"
}
key, url, email = config['key'], config['url'], config['email']
os.environ["ECMWF_API_KEY"] = config['key']
os.environ["ECMWF_API_URL"] = config['url']
os.environ["ECMWF_API_EMAIL"] = config['email']

param_dict = {
    'total_precipitation': '228.128',
    '2m_temperature': '167.128',
    '10m_u_component_of_wind': '165.128',
    '10m_v_component_of_wind': '166.128',
}

def main(
    var_name,
    years=[2018],
    month_start=1,
    month_end=12,
    step_start=0,
    step_end=360,
    step_interval=6,
    save_dir='data/weather_bench/baseline',
):
    save_dir = os.path.join(save_dir, var_name)
    os.makedirs(save_dir, exist_ok=True)

    server = ECMWFService("mars")

    months = range(month_start, month_end+1)
    for year in years:
        for month in months:
            days = calendar.monthrange(year, month)[1]
            month = str(month).zfill(2)
            date = f"{year}-{month}-01/to/{year}-{month}-{days}"
            step = ''.join(str(hour) + '/' for hour in np.arange(step_start, step_end, step_interval))
            step += str(step_end)
            target = f"{var_name}_{year}_{month}.grib"

            retrieve = {
                'class': 'od',
                'date': date,
                'expver': '1',
                'levtype': 'sfc',
                'param': param_dict[var_name],
                'step': step,
                'stream': 'enfo',
                'time': '00:00:00',
                'type': 'cf',
            }
            print(retrieve)

            try:
                server.execute(retrieve, target)
            except APIException:
                print(f'Damaged files {year}-{month}-01/to/{year}-{month}-{days}')
                continue


if __name__ == '__main__':
    Fire(main)
