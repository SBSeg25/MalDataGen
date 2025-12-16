#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2022/06/01'
__last_update__ = '2023/08/03'
__credits__ = ['unknown']

from Engine.exception.MetricCalculationError import MetricCalculationError


class MetricCalculationValueError(MetricCalculationError):
    def __init__(self, metric_name, error_message):
        super().__init__(f"Error computing {metric_name}: {error_message}")
