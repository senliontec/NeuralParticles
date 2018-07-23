#******************************************************************************
#
# HyperSearch: NN Parameter Tuning
# Copyright 2018 Steffen Wiewel
#
# hyper parameter class
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#******************************************************************************

import numpy as np
from random import uniform, randint
from enum import Enum

#---------------------------------------------------------------------------------
class ValueType(Enum):
    Integer = 0
    Float   = 1
#---------------------------------------------------------------------------------
class SearchType(Enum):
    Random = 0
    Linear = 1
    List   = 2

#---------------------------------------------------------------------------------
class HyperParameter(object):
    """ Hyper parameter class that is used in HyperSearch """
    """ Initialize with constructor (use kwargs to setup random/linear/list search parameters) """
    """ Use get_values() to receive a valid list of entries """
    #---------------------------------------------------------------------------------
    def __init__(self, parameter_name, search_type = SearchType.Random, value_type = ValueType.Float, **kwargs):
        assert isinstance(parameter_name, str), "Variable 'parameter_name' must be instance of 'basestring' class"
        self.parameter_name = parameter_name
        assert isinstance(search_type, SearchType), "Variable 'search_type' must be instance of 'SearchType' class"
        self.search_type = search_type
        assert isinstance(value_type, ValueType), "Variable 'value_type' must be instance of 'ValueType' class"
        self.value_type = ValueType(value_type)

        if self.search_type is SearchType.Random:
            self._init_random(kwargs.get("value_range"), kwargs.get("iterations"))
        elif self.search_type is SearchType.Linear:
            self._init_linear(kwargs.get("value_range"), kwargs.get("iterations"))
        elif self.search_type is SearchType.List:
            self._init_list(kwargs.get("values"))
        else:
            assert False, "Search type {} is not implemented yet".format(self.search_type)
        # TODO: add feature to sample from given statistical function
        # e.g.: scipy.stats.expon
        # https://docs.scipy.org/doc/scipy/reference/stats.html

    #---------------------------------------------------------------------------------
    @classmethod
    def parse(cls, arg):
        """ e.g. [learning_rate,random,float,[0.1,0.5],3] """
        """ e.g. [learning_rate,linear,float,[0.1,0.5],3] """
        """ e.g. [learning_rate,list,int,[0,1,2,3]] """
        arg = arg.replace("[","").replace("]","")
        arg = arg.split(',')
        
        # name
        name = arg[0]
        # search type -> hyper_parameter -> SearchType
        search_type = 0 # random
        if arg[1] == "random":
            search_type = 0
        elif arg[1] == "linear":
            search_type = 1
        elif arg[1] == "list":
            search_type = 2
        # value type -> hyper_parameter -> ValueType
        val_type = 1 # float
        if arg[2] == "int":
            val_type = 0
        elif arg[2] == "float":
            val_type = 1

        # parse sequence
        if search_type == 0 or search_type == 1:
            # value range
            val_range = []
            val_range.append(float(arg[3]))
            val_range.append(float(arg[4]))

            # iterations
            iterations = int(arg[5])
                        
            return cls(parameter_name=name, search_type=SearchType(search_type), value_type=ValueType(val_type), value_range=val_range, iterations=iterations)
        # parse list
        else:
            values = []
            for val in arg[3:]:
                values.append(float(val))
                    
            return cls(parameter_name=name, search_type=SearchType(search_type), value_type=ValueType(val_type), values=values)

    #---------------------------------------------------------------------------------
    def _init_random(self, value_range, iterations):
        assert isinstance(value_range, list) and len(value_range) == 2 and value_range[0] < value_range[1], "Variable 'value_range' must be instance of 'list' class with two entries"
        self.value_range = value_range
        assert iterations > 0, "There must be at least one iteration"
        self.iterations = iterations

    #---------------------------------------------------------------------------------
    def _init_linear(self, value_range, iterations):
        assert isinstance(value_range, list) and len(value_range) == 2 and value_range[0] < value_range[1], "Variable 'value_range' must be instance of 'list' class with two entries"
        self.value_range = value_range
        assert iterations > 0, "There must be at least one iteration"
        self.iterations = iterations

     #---------------------------------------------------------------------------------
    def _init_list(self, values):
        assert isinstance(values, list) and len(values) > 0, "There must be at least one entry in 'values'"
        self.values = values
        if self.value_type == ValueType.Integer:
            self.values = [int(i) for i in self.values]

    #---------------------------------------------------------------------------------
    def get_values(self):
        """ Returns a list of values """
        if self.search_type is SearchType.Random:
            return self._get_values_random()
        elif self.search_type is SearchType.Linear:
            return self._get_values_linear()
        elif self.search_type is SearchType.List:
            return self.values
        assert False, "Search type {} is not implemented yet".format(self.search_type)
        return None

    #---------------------------------------------------------------------------------
    def _get_value_linear(self, interpolant):
        """ Returns the linear interpolation of the given value range """
        t = np.clip(interpolant, 0.0, 1.0)
        a = self.value_range[0]
        b = self.value_range[1]
        res = a * (1.0 - t) + b * t
        if self.value_type == ValueType.Integer:
            res = int(round(res))
        return res

    #---------------------------------------------------------------------------------
    def _get_values_linear(self):
        """ Returns the linear representation of the given value range """
        a = self.value_range[0]
        b = self.value_range[1]
        res = np.linspace(a, b, num=self.iterations)
        if self.value_type == ValueType.Integer:
            res = np.round(res).astype(int)
        res = np.unique(res)
        return res.tolist()

    #---------------------------------------------------------------------------------
    def _get_value_random(self):
        """ Returns a random sample in the given value range """
        # TODO: add feature to sample from given statistical function
        a = self.value_range[0]
        b = self.value_range[1]
        if self.value_type == ValueType.Integer:
            res = randint(a,b)
        else:
            res = uniform(a,b)
        return res

    #---------------------------------------------------------------------------------
    def _get_values_random(self):
        """ Returns random samples in the given value range """
        # TODO: add feature to sample from given statistical function
        #res = [self.get_value_random() for x in range(self.iterations)]
        a = self.value_range[0]
        b = self.value_range[1]
        res = np.random.uniform(a, b, self.iterations)

        if self.value_type == ValueType.Integer:
            res = np.round(res).astype(int)

        # try to fill gaps introduced by RNG returning duplicate numbers
        #res = np.array(res)
        res = np.unique(res)
        retry_count = 0
        while res.size < self.iterations:
            retry_count += 1
            res = np.append(res, self.get_value_random())
            res = np.unique(res)
            # break the loop if too many retries occur
            if retry_count > 10:
                break
        res = res.tolist()

        return res

    #--------------------------------------------
    @property
    def name(self):
        """ the name of the hyper parameter """
        return self.parameter_name
