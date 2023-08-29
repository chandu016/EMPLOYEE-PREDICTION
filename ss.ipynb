{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "dd37bb4d",
   "metadata": {},
   "source": [
    "## EMPLOYEE PREDICTION"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "342d89dc",
   "metadata": {},
   "source": [
    "Description:The Employee Future Prediction project aims to predict the likelihood of an employee \n",
    "leaving a company based on various factors such as age, experience, gender, education, \n",
    "city, and payment tier. By analyzing these factors, the project aims to provide insights \n",
    "that can help organizations take proactive measures to retain valuable employees and \n",
    "reduce turnover"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5fbeb2ea",
   "metadata": {},
   "source": [
    "## all needed libraries to be run related to classifier problems"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1609fcda",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "import pandas as pd \n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.ensemble import AdaBoostClassifier\n",
    "from sklearn.ensemble import BaggingClassifier\n",
    "from sklearn.ensemble import ExtraTreesClassifier\n",
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.metrics import accuracy_score,confusion_matrix,precision_score\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.preprocessing import OrdinalEncoder\n",
    "from sklearn.preprocessing import LabelEncoder\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "278801b4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting lightgbm\n",
      "  Obtaining dependency information for lightgbm from https://files.pythonhosted.org/packages/87/0f/7630ee4fea60ebab5b0e3c35df570cb295c91ece537231a38105c0f243e8/lightgbm-4.0.0-py3-none-win_amd64.whl.metadata\n",
      "  Downloading lightgbm-4.0.0-py3-none-win_amd64.whl.metadata (19 kB)\n",
      "Requirement already satisfied: numpy in c:\\users\\91772\\anconda\\lib\\site-packages (from lightgbm) (1.23.5)\n",
      "Requirement already satisfied: scipy in c:\\users\\91772\\anconda\\lib\\site-packages (from lightgbm) (1.10.1)\n",
      "Downloading lightgbm-4.0.0-py3-none-win_amd64.whl (1.3 MB)\n",
      "   ---------------------------------------- 0.0/1.3 MB ? eta -:--:--\n",
      "   --- ------------------------------------ 0.1/1.3 MB 3.5 MB/s eta 0:00:01\n",
      "   --------- ------------------------------ 0.3/1.3 MB 2.1 MB/s eta 0:00:01\n",
      "   --------------------- ------------------ 0.7/1.3 MB 4.0 MB/s eta 0:00:01\n",
      "   --------------------------- ------------ 0.9/1.3 MB 4.1 MB/s eta 0:00:01\n",
      "   ---------------------------------- ----- 1.1/1.3 MB 4.4 MB/s eta 0:00:01\n",
      "   ---------------------------------------  1.3/1.3 MB 3.9 MB/s eta 0:00:01\n",
      "   ---------------------------------------  1.3/1.3 MB 3.9 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 1.3/1.3 MB 3.1 MB/s eta 0:00:00\n",
      "Installing collected packages: lightgbm\n",
      "Successfully installed lightgbm-4.0.0\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "!pip install lightgbm\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "4a3c5a87",
   "metadata": {},
   "outputs": [],
   "source": [
    "from lightgbm import LGBMClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "47360021",
   "metadata": {},
   "outputs": [],
   "source": [
    "## import file\n",
    "\n",
    "data = pd.read_csv(\"Employee.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "b4aca15d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(4653, 9)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "3b1be3a0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 4653 entries, 0 to 4652\n",
      "Data columns (total 9 columns):\n",
      " #   Column                     Non-Null Count  Dtype \n",
      "---  ------                     --------------  ----- \n",
      " 0   Education                  4653 non-null   object\n",
      " 1   JoiningYear                4653 non-null   int64 \n",
      " 2   City                       4653 non-null   object\n",
      " 3   PaymentTier                4653 non-null   int64 \n",
      " 4   Age                        4653 non-null   int64 \n",
      " 5   Gender                     4653 non-null   object\n",
      " 6   EverBenched                4653 non-null   object\n",
      " 7   ExperienceInCurrentDomain  4653 non-null   int64 \n",
      " 8   LeaveOrNot                 4653 non-null   int64 \n",
      "dtypes: int64(5), object(4)\n",
      "memory usage: 327.3+ KB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "5ad47540",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Education                    0\n",
       "JoiningYear                  0\n",
       "City                         0\n",
       "PaymentTier                  0\n",
       "Age                          0\n",
       "Gender                       0\n",
       "EverBenched                  0\n",
       "ExperienceInCurrentDomain    0\n",
       "LeaveOrNot                   0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "311c734e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1889"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "duplicates= data.duplicated().sum()\n",
    "duplicates"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "c22aea82",
   "metadata": {},
   "outputs": [],
   "source": [
    "data1 = data.drop_duplicates()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "3ba469df",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(2764, 9)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data1.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "acd09ef0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Education</th>\n",
       "      <th>JoiningYear</th>\n",
       "      <th>City</th>\n",
       "      <th>PaymentTier</th>\n",
       "      <th>Age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>EverBenched</th>\n",
       "      <th>ExperienceInCurrentDomain</th>\n",
       "      <th>LeaveOrNot</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Bachelors</td>\n",
       "      <td>2017</td>\n",
       "      <td>Bangalore</td>\n",
       "      <td>3</td>\n",
       "      <td>34</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Bachelors</td>\n",
       "      <td>2013</td>\n",
       "      <td>Pune</td>\n",
       "      <td>1</td>\n",
       "      <td>28</td>\n",
       "      <td>Female</td>\n",
       "      <td>No</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Bachelors</td>\n",
       "      <td>2014</td>\n",
       "      <td>New Delhi</td>\n",
       "      <td>3</td>\n",
       "      <td>38</td>\n",
       "      <td>Female</td>\n",
       "      <td>No</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Masters</td>\n",
       "      <td>2016</td>\n",
       "      <td>Bangalore</td>\n",
       "      <td>3</td>\n",
       "      <td>27</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Masters</td>\n",
       "      <td>2017</td>\n",
       "      <td>Pune</td>\n",
       "      <td>3</td>\n",
       "      <td>24</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Education  JoiningYear       City  PaymentTier  Age  Gender EverBenched  \\\n",
       "0  Bachelors         2017  Bangalore            3   34    Male          No   \n",
       "1  Bachelors         2013       Pune            1   28  Female          No   \n",
       "2  Bachelors         2014  New Delhi            3   38  Female          No   \n",
       "3    Masters         2016  Bangalore            3   27    Male          No   \n",
       "4    Masters         2017       Pune            3   24    Male         Yes   \n",
       "\n",
       "   ExperienceInCurrentDomain  LeaveOrNot  \n",
       "0                          0           0  \n",
       "1                          3           1  \n",
       "2                          2           0  \n",
       "3                          5           1  \n",
       "4                          2           1  "
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "008c6221",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>JoiningYear</th>\n",
       "      <th>PaymentTier</th>\n",
       "      <th>Age</th>\n",
       "      <th>ExperienceInCurrentDomain</th>\n",
       "      <th>LeaveOrNot</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>2764.000000</td>\n",
       "      <td>2764.000000</td>\n",
       "      <td>2764.000000</td>\n",
       "      <td>2764.000000</td>\n",
       "      <td>2764.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>2015.090449</td>\n",
       "      <td>2.636035</td>\n",
       "      <td>30.952967</td>\n",
       "      <td>2.644356</td>\n",
       "      <td>0.393632</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1.885943</td>\n",
       "      <td>0.624001</td>\n",
       "      <td>5.108872</td>\n",
       "      <td>1.610610</td>\n",
       "      <td>0.488643</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>2012.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2013.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>27.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>2015.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>30.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>2017.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>2018.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>41.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       JoiningYear  PaymentTier          Age  ExperienceInCurrentDomain  \\\n",
       "count  2764.000000  2764.000000  2764.000000                2764.000000   \n",
       "mean   2015.090449     2.636035    30.952967                   2.644356   \n",
       "std       1.885943     0.624001     5.108872                   1.610610   \n",
       "min    2012.000000     1.000000    22.000000                   0.000000   \n",
       "25%    2013.000000     2.000000    27.000000                   1.000000   \n",
       "50%    2015.000000     3.000000    30.000000                   2.000000   \n",
       "75%    2017.000000     3.000000    35.000000                   4.000000   \n",
       "max    2018.000000     3.000000    41.000000                   7.000000   \n",
       "\n",
       "        LeaveOrNot  \n",
       "count  2764.000000  \n",
       "mean      0.393632  \n",
       "std       0.488643  \n",
       "min       0.000000  \n",
       "25%       0.000000  \n",
       "50%       0.000000  \n",
       "75%       1.000000  \n",
       "max       1.000000  "
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data1.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "b50cb950",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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Y8eOa2oDdQxERERERErRlClTcHNzs3hNmTKl2Lw5OTns3LmT7t27W6R3796dTZs2FVumbdu2nDp1iuXLl1NQUEBsbCwLFy6kT58+11RPTT4WERERESnFycfjxo1j7NixFmn29vbF5k1ISCA/Px9vb2+LdG9vb2JiYoot07ZtW3744QcGDBhAVlYWeXl53HvvvXz66afXVE9FDERERERESpG9vT2urq4Wr5I6BucZLumoFBQUFEk7LywsjFGjRvHaa6+xc+dOVqxYwbFjxxg5cuQ11VMRAxERERERq1tjuVIvLy+sra2LRAfi4uKKRBHOmzJlCu3ateOFF14AoHHjxjg5OdGhQwfeeustfH19r2rfihiIiIiIiNwi7OzsaNGiBatWrbJIX7VqFW3bti22TEZGBlaXPLnZ2toaKIw0XC1FDEREREREDLfO/fKxY8cyZMgQWrZsSUhICF999RVRUVHmoUHjxo3j9OnTfPvttwDcc889PP7443zxxRf06NGD6OhoRo8eTevWralatepV71cdAxERERGRW2QoEcCAAQM4e/YskyZNIjo6mkaNGrF8+XL8/PwAiI6OtnimwbBhw0hNTeWzzz7jueeew93dnTvvvJN33333mvZrKLiW+IKIXJMj7XuUdRUqlFf6DSzrKlQ43m4uZV2FCiUpPbOsq1Dh3FHXr6yrUKE83atjme371KRru4i+FtVfe6nUtn0jKWIgIiIiIlKKy5XeLm6dwVQiIiIiIlJmFDEQKUUa2nJzTV48v6yrUOHMfvKZsq5ChZKVm1fWVahwFm7dW9ZVqFDKciiR4RaafFxW1AIiIiIiIqKIgYiIiIjIrbQqUVlRxEBERERERBQxEBERERHRqkTqGIiIiIiIgJUG0qgFREREREREEQMREREREQ0lUsRARERERERQxEBEREREBIOWK1XEQEREREREFDEQEREREQGD7perBURERERERBEDERERERGtSqSOgYiIiIgIaPKxhhKJiIiIiIgiBiIiIiIimnyMIgYiIiIiIoIiBiIiIiIiesAZihiIiIiIiAiKGIiIiIiIaLlSFDEQEREREREUMRARERERASvdL1fHQEREREREHQMNJRIREREREUUMREREREQ0+RhFDEREREREBEUMRERERET0gDMUMRARERERERQxEBEREREBg+6XqwVEREREREQRAxER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iIiIiCIGIiIiIiJg0P1ytYCIiIiIiKhjICIiIiKClaH0Xtdh+vTp1K5dG6PRSIsWLVi/fv1l82dnZzN+/Hj8/Pywt7enTp06zJo165r2qaFEIiIiIiK30HKlCxYsYPTo0UyfPp127doxY8YMevXqRVhYGDVr1iy2zEMPPURsbCzffPMNAQEBxMXFkZeXd037VcdAREREROQW8uGHHzJ8+HBGjBgBwLRp01i5ciVffPEFU6ZMKZJ/xYoVrF27lqNHj1KpUiUAatWqdc371VAiERERERGDVam9srOzSUlJsXhlZ2cXW42cnBx27txJ9+7dLdK7d+/Opk2bii3z22+/0bJlS9577z2qVatGYGAgzz//PJmZmdfUBOoYiIiIiIiUoilTpuDm5mbxKu7OP0BCQgL5+fl4e3tbpHt7exMTE1NsmaNHj7Jhwwb279/PL7/8wrRp01i4cCFPPfXUNdVTQ4lEREREpMIrzQecjRs3jrFjx1qk2dvbX74+l8x5KCgoKJJ2nslkwmAw8MMPP+Dm5gYUDkd64IEH+Pzzz3FwcLiqeqpjICIiIiJSiuzt7a/YETjPy8sLa2vrItGBuLi4IlGE83x9falWrZq5UwAQFBREQUEBp06dom7dule1bw0lEhERERExGErvdQ3s7Oxo0aIFq1atskhftWoVbdu2LbZMu3btOHPmDGlpaea0w4cPY2VlRfXq1a963+oYiIiIiIjcQsaOHcvMmTOZNWsW4eHhjBkzhqioKEaOHAkUDk0aOnSoOf+gQYPw9PTk0UcfJSwsjHXr1vHCCy/w2GOPXfUwItBQIhERERERsLp17pcPGDCAs2fPMmnSJKKjo2nUqBHLly/Hz88PgOjoaKKiosz5nZ2dWbVqFc888wwtW7bE09OThx56iLfeeuua9quOgYiIiIjILfSAM4Ann3ySJ598stjP5syZUyStfv36RYYfXatbp2skIiIiIiJlRhEDEREREZFSXK70dqGIgYiIiIiIKGIgIiIiImIw6H65WkBERERERBQxEBERERG51VYlKguKGIiIiIiIiCIGIiIiIiJalUgdAxERERER0ORjDSUSERERERFFDERERERENJQIRQxERERERARFDEREREREMGi5UnUMREqTt5tLWVehQpn95DNlXYUK59Hpn5Z1FSqUV/oNLOsqVDi6WJSKRB0DEREREREr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e4oYiIiIiEiFp4U6FTEQERERERHUMRARERERoaCgoNRe12P69OnUrl0bo9FIixYtWL9+/VWV27hxIzY2NjRt2vSa96mOgYiIiIhUeKaC0ntdqwULFjB69GjGjx/P7t276dChA7169SIqKuqy5ZKTkxk6dCh33XXXdbWBOgYiIiIiIqUoOzublJQUi1d2dnaJ+T/88EOGDx/OiBEjCAoKYtq0adSoUYMvvvjisvt54oknGDRoECEhIddVT3UMRERERKTCK82hRFOmTMHNzc3iNWXKlGLrkZOTw86dO+nevbtFevfu3dm0aVOJ9Z89ezaRkZG8/vrr190GWpVIRERERKQUjRs3jrFjx1qk2dvbF5s3ISGB/Px8vL29LdK9vb2JiYkptsyRI0d4+eWXWb9+PTY21395r46BiIiIiFR4JkpvuVJ7e/sSOwIlMRgMFu8LCgqKpAHk5+czaNAgJk6cSGBg4L+qpzoGIiIiIiK3CC8vL6ytrYtEB+Li4opEEQBSU1PZsWMHu3fv5umnnwbAZDJRUFCAjY0Nf/zxB3feeedV7VsdAxERERGp8G6VB5zZ2dnRokULVq1axf33329OX7VqFX379i2S39XVlX379lmkTZ8+nb/++ouFCxdSu3btq963OgYiIiIiIreQsWPHMmTIEFq2bElISAhfffUVUVFRjBw5Eiics3D69Gm+/fZbrKysaNSokUX5KlWqYDQai6RfiToGIiIiIlLh3SIBAwAGDBjA2bNnmTRpEtHR0TRq1Ijly5fj5+cHQHR09BWfaXA9DAW3StxEpBx6+puFZV2FCsXVwVjWVahwHp3+aVlXoUJ5pd/Asq5ChZOQmlHWVahQ/n79qTLb99HouFLbtr9vlVLb9o2k5xiIiIiIiIiGEomIiIiIaBCNIgYiIiIiIoIiBiIiIiIimBQxUMRAREREREQUMRARERERuaWWKy0rihiIiIiIiIgiBiIiIiIiWpVIHQMREREREU0+RkOJREREREQERQxERERERDSUCEUMREREREQERQxERERERFC8QBEDERERERFBEQMREREREa1KhCIGIiIiIiKCIgYiIiIiIlqVCHUMREREREQ0lAgNJRIRERERERQxEBERERFBAQNFDEREREREBEUMREREREQ0+Rh1DERKVVJ6ZllXoULJys0r6ypUOK/0G1jWVahQJi+eX9ZVqHDeHDi0rKsgctOoYyAiIiIiFZ5WJdIcAxERERERQREDERERERHNMUAdAxERERERTOoXaCiRiIiIiIgoYiAiIiIiQgEKGShiICIiIiIiihiIiIiIiGjysSIGIiIiIiKCIgYiIiIiInrAGYoYiIiIiIgIihiIiIiIiKCAgToGIiIiIiKafIyGEomIiIiICIoYiIiIiIho8jGKGIiIiIiICIoYiIiIiIhojgGKGIiIiIiICIoYiIiIiIhgUsBAEQMREREREVHEQEREREREcwxQx0BERERERB0DNJRIRERERERQxEBEREREBBOKGChiICIiIiIiihiIiIiIiGiKgSIGIiIiIiKCIgYiIiIiIlqVCEUMREREREQERQxERERERDApYqCIgYiIiIiIKGIgIiIiIqI5BqhjICIiIiKCSf0CDSUSERERERFFDERERERENJQIRQxERERERARFDEREREREFDFAEQMREREREUERAxERERERPeCMchQxGDZsGPfdd19ZV0NuUwaDgSVLlpR1NURERETKzDVFDIYNG8bcuXOLpPfo0YMVK1bcsEpdj48//viWGxu2Zs0aunTpQmJiIu7u7lddLiUlhXfffZdFixZx/Phx3N3dadSoEU8++ST3338/BoOh9Cr9L9SqVYvRo0czevRoc9r5NoDCi28XFxf8/f3p1q0bY8aMwdfXt4xqayk6OhoPD4+yrsYN1a9NY7o0rIuT0Y7ImATmrNnG6XPJly3Tqk5NHghpQhU3F+KSU/l5Uyg7jp40f35XcCB3BQdS2dUJgFNnk/ll2172njhjztOyTg3ubBRI7SqVcHEw8sqPS4lKSCydgyxD97ZsRMegOjja23Is7hw/rN/BmcSUy5ZpXrs697UKprKbM/HJafyybS+7j5+2yNO5YQA9mtTH3dGBM4nJzN+4myMx8cVub0jHlnRqEMD8jbtYve+wxWf+3p7c37ox/lU8yTeZOHk2iWnL1pKbn//vDvwW8WBIU7oG18PZaMeR6Hhm/rWFU2eTLlumTV0/BrZtjrebC7HJqczbuJNtEVEW23wopJlFmaT0DB6fscAiT7t6tfF0cSIv38TR2LPM27iTiJiEG3p85YWxSSM8Bj2IsV5dbLw8OTPuDdLXby7rat0W7m8dTOeGATjZ2xEZe5Zv126/4jm8ZZ0a9G/ThCpuzsQlp7FwSyg7j54y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zv/5ubvnl2Q/z9vbm5iYq3si9dSpU0lPT+ehhx66pja4oasSXcmaNWtu5u5ERERERMpccUuaX7qS4aWuZgn34sybN4833niDX3/9lSpVSn4ifHFuasdARERERORWVJqzbotb0rwkXl5eWFtbF4kOxMXFFYkiXGrBggUMHz6cn3/+ma5du15zPTWUSERERETkFmFnZ0eLFi1YtWqVRfqqVato27ZtCaUKIwXDhg3jxx9/pE+fPte1b0UMRERERKTCM91Cy5WOHTuWIUOG0LJlS0JCQvjqq6+Iiopi5MjCJX/HjRvH6dOn+fbbb4HCTsHQoUP5+OOPueOOO8zRBgcHB/NS91dDHQMRERERkVvIgAEDOHv2LJMmTSI6OppGjRqxfPly/Pz8AIiOjiYqKsqcf8aMGeTl5fHUU0/x1FNPmdMfeeQR5syZc9X7VcdARERERCq8Am6diAHAk08+yZNPPlnsZ5de7N+oBX7UMRARERGRCu9WevJxWdHkYxERERERUcRARERERMSkgIEiBiIiIiIiooiBiIiIiIjmGKCIgYiIiIiIoIiBiIiIiIgiBihiICIiIiIiKGIgIiIiIoJJEQN1DERERERE1DHQUCIREREREUERAxERERERTT5GEQMREREREUERAxERERERTAoYKGIgIiIiIiKKGIiIiIiIaI4BihiIiIiIiAiKGIiIiIiIKGKAOgYiIiIiInrAGRpKJCIiIiIiKGIgIiIiIoICBooYiIiIiIgIihiIiIiIiGiOAYoYiIiIiIgIihiIiIiIiFCAIgaKGIiIiIiIiCIGIiIiIiJ6wJk6BiIiIiIimNQv0FAiERERERFRxEBEREREREOJUMRARERERERQxEBERERERA84QxEDERERERFBEQORUnVHXb+yrkKFsnDr3rKuQoVjMBjKugoVypsDh5Z1FSqcCfO/LesqVCxPDy6zXWuOgSIGIiIiIiKCIgYiIiIiIihgoI6BiIiIiIgmH6OhRCIiIiIigiIGIiIiIiKafIwiBiIiIiIigiIGIiIiIiKafIwiBiIiIiIigiIGIiIiIiKYUMhAEQMREREREVHEQEREREREqxKpYyAiIiIiogecoaFEIiIiIiKCIgYiIiIiIlquFEUMREREREQERQxERERERDT5GEUMREREREQERQxERERERLQqEYoYiIiIiIgIihiIiIiI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AHBfq2AAPn60n8X+3lmymhWhB0vzkG558zfuxt7GhtG9C9s8/FQsL3z3m0WbV3Fzsbhz5+nixMyRA8zvB7ZtxsC2zQg9fpoxc5cAUK9qZaYNuxDKf6pH4flnRWg47/76Vykf1a1j2a4w7GyseaRTaxzt7Tgam8B7v/5FVm6eOY+ni5PFdzoiJoHpKzfQ/44m9G/TmLjkNKav3MDR2AvPULnrn+Vkx/ezjAJ/tXozG/5ZIrZZ7er8t2uI+bOnehb+DX7Ztpdftu278Qd7GzPWD6T6p++b31ceVXh3OWX5H8ROnlpSMfnHrfb7/uSTT/Lkk08W+9n51RMv1qlTJ3bt2vWv9mkouNVaQaQc+ex/FfuhSDfbwkueWiul73Z8tsftrIane1lXocKZMP/bsq5ChVJ3w8oy2/fQz34otW1/+/TtEVHXHAMREREREdFQIhERERERDaJRxEBERERERFDEQEREREREEQMUMRARERERERQxEBERERG5rgeRlTeKGIiIiIiIiCIGIiIiIiIKGKhjICIiIiJCAeoZaCiRiIiIiIgoYiAiIiIiosnHihiIiIiIiAiKGIiIiIiI6AFnKGIgIiIiIiIoYiAiIiIigkkBA0UMREREREREEQMREREREc0xQB0DERERERF1DNBQIhERERERQREDERERERE94AxFDEREREREBEUMRERERERQwEARAxERERERQREDERERERHNMUARAxERERERQREDERERERE9xwB1DEREREREKEAdAw0lEhERERERRQxEREREREwKGChiICIiIiIiihiIiIiIiGjyMYoYiIiIiIgIihiIiIiIiChigCIGIiIiIiKCIgYiIiIiIpgUMVDHQERERERE/QINJRIRERERERQxEBERERHRUCIUMRARERERERQxEBERERHRcqUoYiAiIiIiIoChQN0jEblIdnY2U6ZMYdy4cdjb25d1dSoEtfnNpfa+udTeN5/aXK6XOgYiYiElJQU3NzeSk5NxdXUt6+pUCGrzm0vtfXOpvW8+tblcLw0lEhERERERdQxEREREREQdAxERERERQR0DEbmEvb09r7/+uias3URq85tL7X1zqb1vPrW5XC9NPhYREREREUUMREREREREHQMREREREUEdAxERERERQR0DERERERFBHQORCi0vLw8bGxv2799f1lURERGRMqaOgUgF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",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "## check corelation between the variables\n",
    "f, ax = plt.subplots(figsize=(8, 6))\n",
    "corr = data1.corr()\n",
    "sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool),\n",
    "            cmap=sns.diverging_palette(220, 10, as_cmap=True),\n",
    "            square=True, ax=ax ,annot=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "52a75298",
   "metadata": {},
   "source": [
    "## PaymentTier' & 'Age' have a very weak negative correlation with Our Target variable ('LeaveOrNot')\n",
    "## 'JoiningYear' has a very weak positive correlation with Target variable ('LeaveOrNot')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "0e9f245c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\91772\\AppData\\Local\\Temp\\ipykernel_10040\\3290513524.py:2: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  data1['JoiningYear'] = data1['JoiningYear'].astype('object')\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='JoiningYear', ylabel='count'>"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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sWKFAgICfFg5AADwFzZzNd5p5GZFRUUKDw9XYWGhwsLCfF0OAOACHZ553evnzJr/sMvrn3/+WTk5OWrevLlCQkK8Xs+16uzZszp8+LDi4uIqfKGpqn+/ff6cHQAArgZBQUGSpDNnzvi4kmvL+fk+P//V4deXsQAA8BcBAQGqX7++8vPzJUl16tS57KNQUH3GGJ05c0b5+fmqX79+jW5PIewAAFBFDodDkpyBB55Xv35957xXF2EHAIAqstlsaty4saKiolRaWurrciwvKCjILV84IuwAAHCFAgIC+NbvVYQblAEAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKURdgAAgKX5NOy8/PLLuuWWWxQWFqawsDB16dJFf/7zn537jTFKSUlRTEyMQkJClJiYqL1797q8R3FxscaNG6eGDRuqbt26GjhwoI4dO+btoQAAAD/l07DTpEkTzZkzR59++qk+/fRT3XHHHbr77rudgWbevHlasGCBlixZop07d8rhcKhPnz46efKk8z2SkpK0fv16paWladu2bTp16pQGDBigsrIyXw0LAAD4EZsxxvi6iH8VERGh+fPn69FHH1VMTIySkpI0efJkSf9cxYmOjtbcuXM1ZswYFRYWqlGjRlq1apUeeOABSdJ3332n2NhYbdiwQf369avSOYuKihQeHq7CwkKFhYV5bGwAgOrp8MzrXj9n1vyHvX5OXJmq/v32m3t2ysrKlJaWptOnT6tLly7KyclRXl6e+vbt6+xjt9vVs2dPZWZmSpKysrJUWlrq0icmJkYJCQnOPhdTXFysoqIilw0AAFiTz8PO7t27Va9ePdntdj322GNav369brrpJuXl5UmSoqOjXfpHR0c79+Xl5al27dpq0KBBpX0uZvbs2QoPD3dusbGxbh4VAADwFz4POzfccIOys7O1Y8cOPf744xo+fLj27dvn3G+z2Vz6G2MqtF3ocn2mTJmiwsJC53b06NGaDQIAAPgtn4ed2rVrq2XLlurYsaNmz56ttm3b6r/+67/kcDgkqcIKTX5+vnO1x+FwqKSkRAUFBZX2uRi73e78Btj5DQAAWJPPw86FjDEqLi5WXFycHA6H0tPTnftKSkqUkZGhrl27SpI6dOigoKAglz65ubnas2ePsw8AALi2Bfry5M8++6z69++v2NhYnTx5Umlpadq6das2btwom82mpKQkzZo1S/Hx8YqPj9esWbNUp04dDRkyRJIUHh6ukSNHasKECYqMjFRERIQmTpyoNm3aqHfv3r4cGgAA8BM+DTvff/+9hg0bptzcXIWHh+uWW27Rxo0b1adPH0nSpEmTdPbsWY0dO1YFBQXq3LmzNm3apNDQUOd7LFy4UIGBgRo8eLDOnj2rXr16acWKFQoICPDVsAAAgB/xu+fs+ALP2QEA/8ZzdnAxV91zdgAAADyBsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACwt0NcFALj2dHjmda+eL2v+w149HwD/wsoOAACwNMIOAACwNMIOAACwNMIOAACwNG5QBjyEm3ABwD9Ua2Xnjjvu0IkTJyq0FxUV6Y477qhpTQAAAG5TrbCzdetWlZSUVGj/+eef9de//rXGRQEAALjLFV3G+uKLL5z/ed++fcrLy3O+Lisr08aNG3Xddde5rzoAAIAauqKw065dO9lsNtlstotergoJCdHixYvdVhwAAEBNXVHYycnJkTFGLVq00N///nc1atTIua927dqKiopSQECA24sEAACorisKO82aNZMklZeXe6QYAAAAd6v2V88PHDigrVu3Kj8/v0L4mT59eo0LAwAAcIdqhZ1ly5bp8ccfV8OGDeVwOGSz2Zz7bDYbYQcAAPiNaoWd//iP/9ALL7ygyZMnu7seAAAAt6rWc3YKCgp0//33u7sWAAAAt6tW2Ln//vu1adMmd9cCAADgdtW6jNWyZUtNmzZNO3bsUJs2bRQUFOSyf/z48W4pDgAAoKaqFXaWLl2qevXqKSMjQxkZGS77bDYbYQcAAPiNaoWdnJwcd9cBAADgEdW6ZwcAAOBqUa2VnUcfffSS+1977bVqFQMAAOBu1Qo7BQUFLq9LS0u1Z88enThx4qI/EAoAAOAr1Qo769evr9BWXl6usWPHqkWLFjUuCgAAwF3cds9OrVq19PTTT2vhwoXueksAAIAac+sNyocOHdK5c+fc+ZYAAAA1Uq3LWMnJyS6vjTHKzc3VBx98oOHDh7ulMAAAAHeoVtjZtWuXy+tatWqpUaNGevHFFy/7TS0AAABvqlbY2bJli7vrAAAA8IhqhZ3zfvjhB3311Vey2Wxq1aqVGjVq5K66AACAHzoys43Xz9l0+u4aHV+tG5RPnz6tRx99VI0bN9Ztt92mHj16KCYmRiNHjtSZM2dqVBAAAIA7VSvsJCcnKyMjQ3/605904sQJnThxQu+++64yMjI0YcIEd9cIAABQbdW6jLV27Vq9/fbbSkxMdLb96le/UkhIiAYPHqyXX37ZXfUBAADUSLVWds6cOaPo6OgK7VFRUVzGAgAAfqVaYadLly6aMWOGfv75Z2fb2bNn9dxzz6lLly5uKw4AAKCmqnUZa9GiRerfv7+aNGmitm3bymazKTs7W3a7XZs2bXJ3jQAAANVWrbDTpk0b/eMf/9Abb7yhL7/8UsYYPfjggxo6dKhCQkLcXaPPdXjmda+eL2v+w149X00wNwAAf1etsDN79mxFR0dr9OjRLu2vvfaafvjhB02ePNktxQEAANRUtcLOK6+8otWrV1dov/nmm/Xggw8SdgCgGry9UiqxWoprQ7VuUM7Ly1Pjxo0rtDdq1Ei5ubk1LgoAAMBdqhV2YmNjtX379grt27dvV0xMTI2LAgAAcJdqXcYaNWqUkpKSVFpaqjvuuEOS9Je//EWTJk3iCcoAAMCvVCvsTJo0ScePH9fYsWNVUlIiSQoODtbkyZM1ZcoUtxYIAABQE9UKOzabTXPnztW0adO0f/9+hYSEKD4+Xna73d31AQAA1Ei1ws559erVU6dOndxVCwAAgNtV6wZlAACAqwVhBwAAWBphBwAAWBphBwAAWBphBwAAWBphBwAAWJpPw87s2bPVqVMnhYaGKioqSoMGDdJXX33l0scYo5SUFMXExCgkJESJiYnau3evS5/i4mKNGzdODRs2VN26dTVw4EAdO3bMm0MBAAB+yqdhJyMjQ0888YR27Nih9PR0nTt3Tn379tXp06edfebNm6cFCxZoyZIl2rlzpxwOh/r06aOTJ086+yQlJWn9+vVKS0vTtm3bdOrUKQ0YMEBlZWW+GBYAAPAjNXqoYE1t3LjR5XVqaqqioqKUlZWl2267TcYYLVq0SFOnTtW9994rSVq5cqWio6O1evVqjRkzRoWFhVq+fLlWrVql3r17S5LeeOMNxcbG6qOPPlK/fv28Pi4AAOA/fBp2LlRYWChJioiIkCTl5OQoLy9Pffv2dfax2+3q2bOnMjMzNWbMGGVlZam0tNSlT0xMjBISEpSZmXnRsFNcXKzi4mLn66KiIk8NCQBwlToys41Xz9d0+m6vnu9a4jc3KBtjlJycrO7duyshIUGSlJeXJ0mKjo526RsdHe3cl5eXp9q1a6tBgwaV9rnQ7NmzFR4e7txiY2PdPRwAAOAn/CbsPPnkk/riiy/0xz/+scI+m83m8toYU6HtQpfqM2XKFBUWFjq3o0ePVr9wAADg1/wi7IwbN07vvfeetmzZoiZNmjjbHQ6HJFVYocnPz3eu9jgcDpWUlKigoKDSPhey2+0KCwtz2QAAgDX5NOwYY/Tkk09q3bp12rx5s+Li4lz2x8XFyeFwKD093dlWUlKijIwMde3aVZLUoUMHBQUFufTJzc3Vnj17nH0AAMC1y6c3KD/xxBNavXq13n33XYWGhjpXcMLDwxUSEiKbzaakpCTNmjVL8fHxio+P16xZs1SnTh0NGTLE2XfkyJGaMGGCIiMjFRERoYkTJ6pNmzbOb2cBAIBrl0/DzssvvyxJSkxMdGlPTU3ViBEjJEmTJk3S2bNnNXbsWBUUFKhz587atGmTQkNDnf0XLlyowMBADR48WGfPnlWvXr20YsUKBQQEeGsoAADAT/k07BhjLtvHZrMpJSVFKSkplfYJDg7W4sWLtXjxYjdWB8Aq+AoxcG3zixuUAQAAPIWwAwAALI2wAwAALI2wAwAALI2wAwAALI2wAwAALI2wAwAALI2wAwAALI2wAwAALI2wAwAALI2wAwAALI2wAwAALI2wAwAALI2wAwAALI2wAwAALC3Q1wUAV+LIzDZeP2fT6bu9fk4AgPuwsgMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACwt0NcFAHCPIzPbePV8Tafv9ur5AKC6WNkBAACWRtgBAACWRtgBAACWRtgBAACWRtgBAACWRtgBAACWRtgBAACWRtgBAACWRtgBAACWRtgBAACW5tOw8/HHH+uuu+5STEyMbDab3nnnHZf9xhilpKQoJiZGISEhSkxM1N69e136FBcXa9y4cWrYsKHq1q2rgQMH6tixY14cBQAA8Gc+DTunT59W27ZttWTJkovunzdvnhYsWKAlS5Zo586dcjgc6tOnj06ePOnsk5SUpPXr1ystLU3btm3TqVOnNGDAAJWVlXlrGAAAwI/59IdA+/fvr/79+190nzFGixYt0tSpU3XvvfdKklauXKno6GitXr1aY8aMUWFhoZYvX65Vq1apd+/ekqQ33nhDsbGx+uijj9SvX7+LvndxcbGKi4udr4uKitw8MgAA4C/89p6dnJwc5eXlqW/fvs42u92unj17KjMzU5KUlZWl0tJSlz4xMTFKSEhw9rmY2bNnKzw83LnFxsZ6biAAAMCn/Dbs5OXlSZKio6Nd2qOjo5378vLyVLt2bTVo0KDSPhczZcoUFRYWOrejR4+6uXoAAOAvfHoZqypsNpvLa2NMhbYLXa6P3W6X3W53S30AAMC/+e3KjsPhkKQKKzT5+fnO1R6Hw6GSkhIVFBRU2gcAAFzb/DbsxMXFyeFwKD093dlWUlKijIwMde3aVZLUoUMHBQUFufTJzc3Vnj17nH0AAMC1zaeXsU6dOqWDBw86X+fk5Cg7O1sRERFq2rSpkpKSNGvWLMXHxys+Pl6zZs1SnTp1NGTIEElSeHi4Ro4cqQkTJigyMlIRERGaOHGi2rRp4/x2FgAAuLb5NOx8+umnuv32252vk5OTJUnDhw/XihUrNGnSJJ09e1Zjx45VQUGBOnfurE2bNik0NNR5zMKFCxUYGKjBgwfr7Nmz6tWrl1asWKGAgACvjwcAAPgfn4adxMREGWMq3W+z2ZSSkqKUlJRK+wQHB2vx4sVavHixByoEAABXO7+9ZwcAAMAdCDsAAMDS/P45OwAAzzkys41Xz9d0+m6vng+QWNkBAAAWR9gBAACWxmUsP+TtZWWJpWUAgHWxsgMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACyNsAMAACwt0NcFuMtLL72k+fPnKzc3VzfffLMWLVqkHj16+LosAAA8qsMzr3v1fOtDvXo6t7DEys5bb72lpKQkTZ06Vbt27VKPHj3Uv39/HTlyxNelAQAAH7NE2FmwYIFGjhypUaNGqXXr1lq0aJFiY2P18ssv+7o0AADgY1f9ZaySkhJlZWXpt7/9rUt73759lZmZedFjiouLVVxc7HxdWFgoSSoqKrpo/7Lis26qtmpOBpV59XxS5WO/HOamclafm+rOi8TcVMbb8yIxN5fC3FycP/07fL7dGHPpNzBXuW+//dZIMtu3b3dpf+GFF0yrVq0uesyMGTOMJDY2NjY2NjYLbEePHr1kVrjqV3bOs9lsLq+NMRXazpsyZYqSk5Odr8vLy3X8+HFFRkZWeoy3FBUVKTY2VkePHlVYWJhPa/E3zE3lmJvKMTeVY24qx9xcnL/NizFGJ0+eVExMzCX7XfVhp2HDhgoICFBeXp5Le35+vqKjoy96jN1ul91ud2mrX7++p0qslrCwML/4IPkj5qZyzE3lmJvKMTeVY24uzp/mJTw8/LJ9rvoblGvXrq0OHTooPT3dpT09PV1du3b1UVUAAMBfXPUrO5KUnJysYcOGqWPHjurSpYuWLl2qI0eO6LHHHvN1aQAAwMcsEXYeeOAB/fTTT5o5c6Zyc3OVkJCgDRs2qFmzZr4u7YrZ7XbNmDGjwmU2MDeXwtxUjrmpHHNTOebm4q7WebEZc7nvawEAAFy9rvp7dgAAAC6FsAMAACyNsAMAACyNsAMAACyNsOMBs2fPVqdOnRQaGqqoqCgNGjRIX331lUsfY4xSUlIUExOjkJAQJSYmau/evS59li5dqsTERIWFhclms+nEiRMu+w8fPqyRI0cqLi5OISEhuv766zVjxgyVlJR4eojV4q15kaSBAweqadOmCg4OVuPGjTVs2DB99913nhxejXhzbs4rLi5Wu3btZLPZlJ2d7YFRuYc356Z58+ay2Wwu24W/u+dPvP25+eCDD9S5c2eFhISoYcOGuvfeez01tBrz1txs3bq1wmfm/LZz505PD7NavPm5OXDggO6++241bNhQYWFh6tatm7Zs2eLJ4V0UYccDMjIy9MQTT2jHjh1KT0/XuXPn1LdvX50+fdrZZ968eVqwYIGWLFminTt3yuFwqE+fPjp58qSzz5kzZ3TnnXfq2Wefveh5vvzyS5WXl+uVV17R3r17tXDhQv3hD3+otL+veWteJOn222/XmjVr9NVXX2nt2rU6dOiQfvOb33h0fDXhzbk5b9KkSZd9xLo/8PbcnH+Exfntd7/7ncfGVlPenJu1a9dq2LBheuSRR/T5559r+/btGjJkiEfHVxPempuuXbu6fF5yc3M1atQoNW/eXB07dvT4OKvDm5+bX//61zp37pw2b96srKwstWvXTgMGDKjwqwceV7Of4URV5OfnG0kmIyPDGGNMeXm5cTgcZs6cOc4+P//8swkPDzd/+MMfKhy/ZcsWI8kUFBRc9lzz5s0zcXFxbqvdk7w5L++++66x2WympKTEbfV7kqfnZsOGDebGG280e/fuNZLMrl27PDEMj/Dk3DRr1swsXLjQU6V7nKfmprS01Fx33XXm1Vdf9Wj9nuStf29KSkpMVFSUmTlzplvr9yRPzc0PP/xgJJmPP/7Y2VZUVGQkmY8++sgzg6kEKzteUFhYKEmKiIiQJOXk5CgvL099+/Z19rHb7erZs6cyMzNrfK7z5/F33pqX48eP680331TXrl0VFBRUs6K9xJNz8/3332v06NFatWqV6tSp476ivcTTn5u5c+cqMjJS7dq10wsvvOC3l4UvxlNz89lnn+nbb79VrVq11L59ezVu3Fj9+/evcFnDn3nr35v33ntPP/74o0aMGFGjer3JU3MTGRmp1q1b6/XXX9fp06d17tw5vfLKK4qOjlaHDh3cO4jLIOx4mDFGycnJ6t69uxISEiTJuXx34Q+VRkdH12hp79ChQ1q8ePFV8TMZ3piXyZMnq27duoqMjNSRI0f07rvv1rxwL/Dk3BhjNGLECD322GN+u8R+KZ7+3Dz11FNKS0vTli1b9OSTT2rRokUaO3ase4r3ME/Ozddffy1JSklJ0e9+9zu9//77atCggXr27Knjx4+7aQSe481/h5cvX65+/fopNja2+gV7kSfnxmazKT09Xbt27VJoaKiCg4O1cOFCbdy40es/vm2Jn4vwZ08++aS++OILbdu2rcI+m83m8toYU6Gtqr777jvdeeeduv/++zVq1KhqvYc3eWNennnmGY0cOVLffPONnnvuOT388MN6//33qz3H3uLJuVm8eLGKioo0ZcqUGtfpC57+3Dz99NPO/3zLLbeoQYMG+s1vfuNc7fFnnpyb8vJySdLUqVN13333SZJSU1PVpEkT/c///I/GjBlTg8o9z1v/Dh87dkwffvih1qxZU63jfcGTc2OM0dixYxUVFaW//vWvCgkJ0auvvqoBAwZo586daty4cY3rrypWdjxo3Lhxeu+997RlyxY1adLE2e5wOCSpQkLOz8+vkKSr4rvvvtPtt9/u/BFUf+eteWnYsKFatWqlPn36KC0tTRs2bNCOHTtqVryHeXpuNm/erB07dshutyswMFAtW7aUJHXs2FHDhw93wwg8x1ufm3/1y1/+UpJ08ODBGr2Pp3l6bs7/UbrpppucbXa7XS1atNCRI0dqUrrHefNzk5qaqsjISA0cOLD6BXuRN/69ef/995WWlqZu3brp1ltv1UsvvaSQkBCtXLnSPYOoIsKOBxhj9OSTT2rdunXavHmz4uLiXPbHxcXJ4XAoPT3d2VZSUqKMjAx17dr1is717bffKjExUbfeeqtSU1NVq5b//lfqzXm52Lmlf37d2h95a25+//vf6/PPP1d2drays7O1YcMGSdJbb72lF154wT2DcTNffm527dolSV79f6BXwltz06FDB9ntdpevJ5eWlurw4cN++4PL3v7cGGOUmpqqhx9+2O/vDfTW3Jw5c0aSKvxdqlWrlnO10Gu8div0NeTxxx834eHhZuvWrSY3N9e5nTlzxtlnzpw5Jjw83Kxbt87s3r3bPPTQQ6Zx48amqKjI2Sc3N9fs2rXLLFu2zHlH+65du8xPP/1kjDHm22+/NS1btjR33HGHOXbsmMu5/JG35uWTTz4xixcvNrt27TKHDx82mzdvNt27dzfXX3+9+fnnn70+7qrw1txcKCcnx++/jeWtucnMzDQLFiwwu3btMl9//bV56623TExMjBk4cKDXx1xV3vzcPPXUU+a6664zH374ofnyyy/NyJEjTVRUlDl+/LhXx1xV3v7f1EcffWQkmX379nltjNXlrbn54YcfTGRkpLn33ntNdna2+eqrr8zEiRNNUFCQyc7O9uqYCTseIOmiW2pqqrNPeXm5mTFjhnE4HMZut5vbbrvN7N692+V9ZsyYccn3SU1NrfRc/shb8/LFF1+Y22+/3URERBi73W6aN29uHnvsMXPs2DEvjvbKeGtuLnQ1hB1vzU1WVpbp3LmzCQ8PN8HBweaGG24wM2bMMKdPn/biaK+MNz83JSUlZsKECSYqKsqEhoaa3r17mz179nhppFfO2/+beuihh0zXrl29MLKa8+bc7Ny50/Tt29dERESY0NBQ88tf/tJs2LDBSyP9fzZj/m99HwAAwIL89wYPAAAANyDsAAAASyPsAAAASyPsAAAASyPsAAAASyPsAAAASyPsAAAASyPsAAAASyPsAPCZrVu3ymaz6cSJE1U+ZsSIERo0aJDHagJgPTxBGYBbjRgxQidOnNA777xz2b4lJSU6fvy4oqOjZbPZqvT+hYWFMsaofv36NSv0/xw4cEDt2rXTq6++qiFDhjjby8vL1b17d0VHR2v9+vVuORcA32BlB4DP1K5dWw6Ho8pBR5LCw8PdFnQkqVWrVpozZ47GjRun3NxcZ/uLL76ogwcP6pVXXnHbuc4rLS11+3sCqBxhB4DHFBcXa/z48YqKilJwcLC6d++unTt3OvdfeBlrxYoVql+/vj788EO1bt1a9erV05133ukSQi68jJWYmKjx48dr0qRJioiIkMPhUEpKiksdX375pbp3767g4GDddNNN+uijj2Sz2ZyrT+PGjVO7du00evRoZ//p06dr6dKlioqKUmpqqlq3bq3g4GDdeOONeumll1zef/LkyWrVqpXq1KmjFi1aaNq0aS6BJiUlRe3atdNrr72mFi1ayG63i0V1wHsCfV0AAOuaNGmS1q5dq5UrV6pZs2aaN2+e+vXrp4MHDyoiIuKix5w5c0b/+Z//qVWrVqlWrVr6t3/7N02cOFFvvvlmpedZuXKlkpOT9cknn+hvf/ubRowYoW7duqlPnz4qLy/XoEGD1LRpU33yySc6efKkJkyY4HK8zWZTamqq2rRpo2XLlmn58uV64IEHNGjQIC1btkwzZszQkiVL1L59e+3atUujR49W3bp1NXz4cElSaGioVqxYoZiYGO3evVujR49WaGioJk2a5DzHwYMHtWbNGq1du1YBAQFumF0AVeb131kHYGnDhw83d999tzl16pQJCgoyb775pnNfSUmJiYmJMfPmzTPGGLNlyxYjyRQUFBhjjElNTTWSzMGDB53H/Pd//7eJjo6u8P7n9ezZ03Tv3t2lhk6dOpnJkycbY4z585//bAIDA01ubq5zf3p6upFk1q9f73Lca6+9ZmrVqmViY2PNiRMnjDHGxMbGmtWrV7v0e/75502XLl0qnYN58+aZDh06OF/PmDHDBAUFmfz8/EqPAeA5rOwA8IhDhw6ptLRU3bp1c7YFBQXpF7/4hfbv31/pcXXq1NH111/vfN24cWPl5+df8ly33HKLy+t/Pearr75SbGysHA6Hc/8vfvGLi77PI488omnTpmn8+PEKDw/XDz/8oKNHj2rkyJHOS1ySdO7cOYWHhztfv/3221q0aJEOHjyoU6dO6dy5cwoLC3N572bNmqlRo0aXHAcAzyDsAPAI83/3pFx487Ex5pI3JAcFBbm8ttlsl72/5WLHlJeXV+l8FwoMDFRg4D//aTz/HsuWLVPnzp1d+p2/FLVjxw49+OCDeu6559SvXz+Fh4crLS1NL774okv/unXrVrkGAO7FDcoAPKJly5aqXbu2tm3b5mwrLS3Vp59+qtatW3utjhtvvFFHjhzR999/72z715ukLyU6OlrXXXedvv76a7Vs2dJli4uLkyRt375dzZo109SpU9WxY0fFx8frm2++8chYAFQPKzsAPKJu3bp6/PHH9cwzzygiIkJNmzbVvHnzdObMGY0cOdJrdfTp00fXX3+9hg8frnnz5unkyZOaOnWqpIqrTheTkpKi8ePHKywsTP3791dxcbE+/fRTFRQUKDk5WS1bttSRI0eUlpamTp066YMPPuC5PICfYWUHgFuVl5c7LwPNmTNH9913n4YNG6Zbb71VBw8e1IcffqgGDRp4rZ6AgAC98847OnXqlDp16qRRo0bpd7/7nSQpODj4ssePGjVKr776qlasWKE2bdqoZ8+eWrFihXNl5+6779bTTz+tJ598Uu3atVNmZqamTZvm0TEBuDI8QRmAW915551q2bKllixZ4utSKrV9+3Z1795dBw8edLkZGoA1cRkLgFsUFBQoMzNTW7du1WOPPebrclysX79e9erVU3x8vA4ePKinnnpK3bp1I+gA1wjCDgC3ePTRR7Vz505NmDBBd999t6/LcXHy5ElNmjRJR48eVcOGDdW7d+8K35YCYF1cxgIAAJbGDcoAAMDSCDsAAMDSCDsAAMDSCDsAAMDSCDsAAMDSCDsAAMDSCDsAAMDSCDsAAMDS/hedYSYQLT1s7gAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "## by this we can know that how many employees not leaved and leaved as per the years\n",
    "data1['JoiningYear'] = data1['JoiningYear'].astype('object')\n",
    "sns.countplot(data = data1 ,x='JoiningYear',hue='LeaveOrNot')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9a5abf43",
   "metadata": {},
   "source": [
    "# in 2018 majority of employess leaved the company"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "7fdd9804",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='EverBenched', ylabel='count'>"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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v379ruvcnnnmGXXu3Flt27Z1G+/QoUOlc8jKylLbtm3NwFa/fv1amy3ikhkAALXI5XIpJyfHbZkwYYK++eYbjR49Wvv27dMnn3yibdu26b777lNFRYWaNGmikJAQrVixQidOnNDbb7+tKVOmVNp3QECAfv/732v27Nk6evSo4uLizHVt27ZVfHy8xowZo40bNyo3N1fvvfee5s+fr9dff13Sd4Fo+PDhGjt2rFatWqVPP/1UBw4c0IQJE7R582b97//+70/OCv1YdHS04uPjtXTpUrfxpKQkvfXWW3rqqad0/PhxrVmzRsuWLdPUqVPNmlatWmnnzp36/PPP3b5xVxMIRAAA1KIdO3aoS5cubsucOXO0e/duVVRUaMCAAYqKitKjjz4qh8OhevXqqV69ekpLS1N2draioqL02GOPacGCBZfdf3x8vP7973/rjjvuUIsWLdzWrV69WmPGjFFSUpLatWunYcOGae/evYqIiJD03f1Br776qmbNmqXFixfrlltu0R133KHPPvtM27dvr/JDJZ966qlKl8O6du2qV199VWlpaYqKitKcOXM0b948t0tr8+bN06effqpf/OIXV71XqTrYDE9esKtDioqK5HA45HK5FBQUVGPH4Wv33oOv3QO4Xt9++61yc3PVunVrNWjQwNPtWMbV3vdr/fxmhggAAFgegQgAAFgegQgAAFgegQgAAFgegQgAAFgegQgAAFieRwPRzp07NXToUDmdTtlsNm3atOmKtRMmTJDNZtOSJUvcxktLSzVp0iQ1bdpUAQEBGjZsmE6fPu1WU1hYqISEBDkcDjkcDiUkJOjs2bPVf0IAAKBO8mggOn/+vDp16qRly5ZdtW7Tpk3au3ev+eu8P5SYmKj09HSlpaVp165dKi4u1pAhQ9we9R0XF6ecnBxt2bJFW7ZsUU5OjtvjzAEAgLV59LfMBg0apEGDBl215vPPP9cjjzyirVu36ne/+53bOpfLpVWrVmnt2rXq27evJGndunWKiIjQm2++qQEDBujo0aPasmWL9uzZo27dukmSVq5cqe7du+vYsWNq167dZY9bWlqq0tJS83VRUdGNnCoAAPBiXv3jrhcvXlRCQoKmTZumjh07VlqfnZ2t8vJy9e/f3xxzOp2KiopSVlaWBgwYoHfffVcOh8MMQ5L0m9/8Rg6HQ1lZWVcMRCkpKXryySer/6QAAKii2v41g6o+sf/555/XggULlJeXp44dO2rJkiVX/EFZb+HVN1XPnz9fvr6+mjx58mXX5+fnq379+mrSpInbeFhYmPLz882a0NDQStuGhoaaNZczc+ZMuVwuczl16tQNnAkAANbwyiuvKDExUbNmzdIHH3ygO+64Q4MGDdLJkyc93dpVeW0gys7O1nPPPafU1FTZbLbr2tYwDLdtLrf9j2t+zG63KygoyG0BAABXt2jRIo0fP17333+/2rdvryVLligiIkLLly/3dGtX5bWB6J133lFBQYFatGghX19f+fr66rPPPlNSUpJatWolSQoPD1dZWZkKCwvdti0oKFBYWJhZ8+WXX1ba/5kzZ8waAABw48rKypSdne12K4sk9e/fX1lZWR7q6tp4bSBKSEjQgQMHlJOTYy5Op1PTpk3T1q1bJUkxMTHy8/NTRkaGuV1eXp4OHTqkHj16SJK6d+8ul8ulffv2mTV79+6Vy+UyawAAwI376quvVFFRUWnC4Ye3sngrj95UXVxcrBMnTpivc3NzlZOTo+DgYLVo0UIhISFu9X5+fgoPDzdvhHY4HBo/frySkpIUEhKi4OBgTZ06VdHR0ea3ztq3b6+BAwfqgQce0AsvvCBJevDBBzVkyJAr3lANAACq7se3pPzUbSrewKOBaP/+/brrrrvM11OmTJEkjR07Vqmpqde0j8WLF8vX11exsbEqKSlRnz59lJqaKh8fH7Pm5Zdf1uTJk80pvGHDhv3ks48AAMD1adq0qXx8fCrNBv3wVhZv5dFA1Lt3bxmGcc31n376aaWxBg0aaOnSpVq6dOkVtwsODta6deuq0iIAALhG9evXV0xMjDIyMjR8+HBzPCMjQ7///e892NlP8+rnEAEAgLplypQpSkhI0G233abu3btrxYoVOnnypB566CFPt3ZVBCIAAFBtRo4cqa+//lrz5s1TXl6eoqKi9Prrr6tly5aebu2qCEQAANQRVX1ydG2bOHGiJk6c6Ok2rovXfu0eAACgthCIAACA5RGIAACA5RGIAACA5RGIAACA5RGIAACA5RGIAACA5RGIAACA5RGIAACA5RGIAACA5fHTHQAA1BEn50XX6vFazDl43dvs3LlTCxYsUHZ2tvLy8pSenq677767+purZswQAQCAanP+/Hl16tRJy5Yt83Qr14UZIgAAUG0GDRqkQYMGebqN68YMEQAAsDwCEQAAsDwCEQAAsDwCEQAAsDwCEQAAsDy+ZQYAAKpNcXGxTpw4Yb7Ozc1VTk6OgoOD1aJFCw92dnUEIgAAUG3279+vu+66y3w9ZcoUSdLYsWOVmprqoa5+GoEIAIA6oipPjq5tvXv3lmEYnm7junEPEQAAsDwCEQAAsDwCEQAAsDwCEQAAsDwCEQAA1awu3lRcl1XH+00gAgCgmvj5+UmSLly44OFOrOXS+33p/a8KvnYPAEA18fHxUePGjVVQUCBJatiwoWw2m4e7+vkyDEMXLlxQQUGBGjduLB8fnyrvi0AEAEA1Cg8PlyQzFKHmNW7c2Hzfq4pABABANbLZbGrevLlCQ0NVXl7u6XZ+9vz8/G5oZugSAhEAADXAx8enWj6oUTu4qRoAAFgegQgAAFgegQgAAFieRwPRzp07NXToUDmdTtlsNm3atMlcV15erhkzZig6OloBAQFyOp0aM2aMvvjiC7d9lJaWatKkSWratKkCAgI0bNgwnT592q2msLBQCQkJcjgccjgcSkhI0NmzZ2vhDAEAQF3g0UB0/vx5derUScuWLau07sKFC3r//fc1e/Zsvf/++9q4caOOHz+uYcOGudUlJiYqPT1daWlp2rVrl4qLizVkyBBVVFSYNXFxccrJydGWLVu0ZcsW5eTkKCEhocbPDwAA1A02w0ueL26z2ZSenq677777ijXvvfeefv3rX+uzzz5TixYt5HK51KxZM61du1YjR46UJH3xxReKiIjQ66+/rgEDBujo0aPq0KGD9uzZo27dukmS9uzZo+7du+vDDz9Uu3btLnus0tJSlZaWmq+LiooUEREhl8uloKCg6jvxH4mZ9lKN7RvXJ3vBGE+3AAC4QUVFRXI4HD/5+V2n7iFyuVyy2Wxq3LixJCk7O1vl5eXq37+/WeN0OhUVFaWsrCxJ0rvvviuHw2GGIUn6zW9+I4fDYdZcTkpKinmJzeFwKCIiomZOCgAAeFydCUTffvutHn/8ccXFxZkJLz8/X/Xr11eTJk3casPCwpSfn2/WhIaGVtpfaGioWXM5M2fOlMvlMpdTp05V49kAAABvUicezFheXq5Ro0bp4sWLev7553+y3jAMt9+OudzvyPy45sfsdrvsdnvVGgYAAHWK188QlZeXKzY2Vrm5ucrIyHC7/hceHq6ysjIVFha6bVNQUKCwsDCz5ssvv6y03zNnzpg1AADA2rw6EF0KQx999JHefPNNhYSEuK2PiYmRn5+fMjIyzLG8vDwdOnRIPXr0kCR1795dLpdL+/btM2v27t0rl8tl1gAAAGvz6CWz4uJinThxwnydm5urnJwcBQcHy+l06o9//KPef/99/b//9/9UUVFh3vMTHBys+vXry+FwaPz48UpKSlJISIiCg4M1depURUdHq2/fvpKk9u3ba+DAgXrggQf0wgsvSJIefPBBDRky5IrfMAMAANbi0UC0f/9+3XXXXebrKVOmSJLGjh2ruXPnavPmzZKkzp07u223fft29e7dW5K0ePFi+fr6KjY2ViUlJerTp49SU1PdflDv5Zdf1uTJk81vow0bNuyyzz4CAADW5DXPIfJ21/ocgxvFc4i8B88hAoC672f5HCIAAICaQCACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACW59FAtHPnTg0dOlROp1M2m02bNm1yW28YhubOnSun0yl/f3/17t1bhw8fdqspLS3VpEmT1LRpUwUEBGjYsGE6ffq0W01hYaESEhLkcDjkcDiUkJCgs2fP1vDZAQCAusKjgej8+fPq1KmTli1bdtn1zz77rBYtWqRly5bpvffeU3h4uPr166dz586ZNYmJiUpPT1daWpp27dql4uJiDRkyRBUVFWZNXFyccnJytGXLFm3ZskU5OTlKSEio8fMDAAB1g80wDMPTTUiSzWZTenq67r77bknfzQ45nU4lJiZqxowZkr6bDQoLC9P8+fM1YcIEuVwuNWvWTGvXrtXIkSMlSV988YUiIiL0+uuva8CAATp69Kg6dOigPXv2qFu3bpKkPXv2qHv37vrwww/Vrl27a+qvqKhIDodDLpdLQUFB1f8GfC9m2ks1tm9cn+wFYzzdAgDgBl3r57fX3kOUm5ur/Px89e/f3xyz2+3q1auXsrKyJEnZ2dkqLy93q3E6nYqKijJr3n33XTkcDjMMSdJvfvMbORwOs+ZySktLVVRU5LYAAICfJ68NRPn5+ZKksLAwt/GwsDBzXX5+vurXr68mTZpctSY0NLTS/kNDQ82ay0lJSTHvOXI4HIqIiLih8wEAAN7LawPRJTabze21YRiVxn7sxzWXq/+p/cycOVMul8tcTp06dZ2dAwCAusJrA1F4eLgkVZrFKSgoMGeNwsPDVVZWpsLCwqvWfPnll5X2f+bMmUqzTz9kt9sVFBTktgAAgJ8nrw1ErVu3Vnh4uDIyMsyxsrIyZWZmqkePHpKkmJgY+fn5udXk5eXp0KFDZk337t3lcrm0b98+s2bv3r1yuVxmDQAAsDZfTx68uLhYJ06cMF/n5uYqJydHwcHBatGihRITE5WcnKzIyEhFRkYqOTlZDRs2VFxcnCTJ4XBo/PjxSkpKUkhIiIKDgzV16lRFR0erb9++kqT27dtr4MCBeuCBB/TCCy9Ikh588EENGTLkmr9hBgAAft48Goj279+vu+66y3w9ZcoUSdLYsWOVmpqq6dOnq6SkRBMnTlRhYaG6deumbdu2KTAw0Nxm8eLF8vX1VWxsrEpKStSnTx+lpqbKx8fHrHn55Zc1efJk89tow4YNu+KzjwAAgPV4zXOIvB3PIbIenkMEAHVfnX8OEQAAQG0hEAEAAMsjEAEAAMsjEAEAAMsjEAEAAMsjEAEAAMsjEAEAAMsjEAEAAMsjEAEAAMsjEAEAAMvz6G+ZAd7s5LxoT7eA77WYc9DTLQD4mWOGCAAAWB6BCAAAWB6BCAAAWB6BCAAAWB6BCAAAWB6BCAAAWB6BCAAAWB6BCAAAWB6BCAAAWB6BCAAAWB6BCAAAWB6BCAAAWB6BCAAAWB6BCAAAWB6BCAAAWB6BCAAAWB6BCAAAWB6BCAAAWB6BCAAAWF6VAtFvf/tbnT17ttJ4UVGRfvvb395oTwAAALWqSoFox44dKisrqzT+7bff6p133rnhpgAAAGqT7/UUHzhwwPznI0eOKD8/33xdUVGhLVu26Kabbqq+7gAAAGrBdQWizp07y2azyWazXfbSmL+/v5YuXVptzQEAANSG6wpEubm5MgxDbdq00b59+9SsWTNzXf369RUaGiofH59qbxIAAKAmXVcgatmypSTp4sWLNdIMAACAJ1xXIPqh48ePa8eOHSooKKgUkObMmXPDjQEAANSWKn3LbOXKlerQoYPmzJmj1157Tenp6eayadOmamvuP//5j5544gm1bt1a/v7+atOmjebNm+cWwAzD0Ny5c+V0OuXv76/evXvr8OHDbvspLS3VpEmT1LRpUwUEBGjYsGE6ffp0tfUJAADqtirNED399NP661//qhkzZlR3P27mz5+vv/3tb1qzZo06duyo/fv3695775XD4dCjjz4qSXr22We1aNEipaamqm3btnr66afVr18/HTt2TIGBgZKkxMRE/fOf/1RaWppCQkKUlJSkIUOGKDs7m3ueAABA1QJRYWGhRowYUd29VPLuu+/q97//vX73u99Jklq1aqW///3v2r9/v6TvZoeWLFmiWbNm6Z577pEkrVmzRmFhYVq/fr0mTJggl8ulVatWae3aterbt68kad26dYqIiNCbb76pAQMG1Ph5AAAA71alS2YjRozQtm3bqruXSm6//Xa99dZbOn78uCTp3//+t3bt2qXBgwdL+u5bb/n5+erfv7+5jd1uV69evZSVlSVJys7OVnl5uVuN0+lUVFSUWXM5paWlKioqclsAAMDPU5VmiH75y19q9uzZ2rNnj6Kjo+Xn5+e2fvLkydXS3IwZM+RyuXTLLbfIx8dHFRUV+utf/6rRo0dLkvlgyLCwMLftwsLC9Nlnn5k19evXV5MmTSrV/PDBkj+WkpKiJ598slrOAwAAeLcqBaIVK1aoUaNGyszMVGZmpts6m81WbYHolVde0bp167R+/Xp17NhROTk5SkxMlNPp1NixY92O+UOGYVQa+7Gfqpk5c6amTJlivi4qKlJEREQVzwQAAHizKgWi3Nzc6u7jsqZNm6bHH39co0aNkiRFR0frs88+U0pKisaOHavw8HBJ380CNW/e3NyuoKDAnDUKDw9XWVmZCgsL3WaJCgoK1KNHjyse2263y26318RpAQAAL1Ole4hqy4ULF1SvnnuLPj4+5tfuW7durfDwcGVkZJjry8rKlJmZaYadmJgY+fn5udXk5eXp0KFDVw1EAADAOqo0Q3Tfffdddf2LL75YpWZ+bOjQofrrX/+qFi1aqGPHjvrggw+0aNEi8/g2m02JiYlKTk5WZGSkIiMjlZycrIYNGyouLk6S5HA4NH78eCUlJSkkJETBwcGaOnWqoqOjzW+dAQAAa6vy1+5/qLy8XIcOHdLZs2cv+6OvVbV06VLNnj1bEydOVEFBgZxOpyZMmOD2JOzp06erpKREEydOVGFhobp166Zt27aZzyCSpMWLF8vX11exsbEqKSlRnz59lJqayjOIAACAJMlmGIZRHTu6ePGiJk6cqDZt2mj69OnVsUuvUlRUJIfDIZfLpaCgoBo7Tsy0l2ps37g+6YELPN0CvtdizkFPtwCgjrrWz+9qu4eoXr16euyxx7R48eLq2iUAAECtqNabqj/++GP95z//qc5dAgAA1Lgq3UP0w+fzSN890ycvL0//+te/3J4PBAAAUBdUKRB98MEHbq/r1aunZs2aaeHChT/5DTQAAABvU6VAtH379uruAwAAwGOqFIguOXPmjI4dOyabzaa2bduqWbNm1dUXAABAranSTdXnz5/Xfffdp+bNm+vOO+/UHXfcIafTqfHjx+vChQvV3SMAAECNqlIgmjJlijIzM/XPf/5TZ8+e1dmzZ/WPf/xDmZmZSkpKqu4eAQAAalSVLplt2LBBr732mnr37m2ODR48WP7+/oqNjdXy5curqz8AAIAaV6UZogsXLpi/Jv9DoaGhXDIDAAB1TpUCUffu3fWXv/xF3377rTlWUlKiJ598Ut27d6+25gAAAGpDlS6ZLVmyRIMGDdLNN9+sTp06yWazKScnR3a7Xdu2bavuHgEAAGpUlQJRdHS0PvroI61bt04ffvihDMPQqFGjFB8fL39//+ruEQAAoEZVKRClpKQoLCxMDzzwgNv4iy++qDNnzmjGjBnV0hwAAEBtqNI9RC+88IJuueWWSuMdO3bU3/72txtuCgAAoDZVKRDl5+erefPmlcabNWumvLy8G24KAACgNlUpEEVERGj37t2Vxnfv3i2n03nDTQEAANSmKt1DdP/99ysxMVHl5eX67W9/K0l66623NH36dJ5UDQAA6pwqBaLp06frm2++0cSJE1VWViZJatCggWbMmKGZM2dWa4MAAAA1rUqByGazaf78+Zo9e7aOHj0qf39/RUZGym63V3d/AAAANa5KgeiSRo0a6Ve/+lV19QIAAOARVbqpGgAA4OeEQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACyPQAQAACzP6wPR559/rj/96U8K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wM+cJe4hAgAAuBoCEQAAsDwCEQAAsDwCEQAAsDwCESxr3LhxstlseuaZZ9zGN23axNPIgTrIMAz17dtXAwYMqLTu+eefl8Ph0MmTJz3QGeoCAhEsrUGDBpo/f74KCws93QqAG2Sz2bR69Wrt3btXL7zwgjmem5urGTNm6LnnnlOLFi082CG8GYEIlta3b1+Fh4crJSXlijUbNmxQx44dZbfb1apVKy1cuLAWOwRwPSIiIvTcc89p6tSpys3NlWEYGj9+vPr06aNf//rXGjx4sBo1aqSwsDAlJCToq6++Mrd97bXXFB0dLX9/f4WEhKhv3746f/68B88GtYlABEvz8fFRcnKyli5dqtOnT1dan52drdjYWI0aNUoHDx7U3LlzNXv2bKWmptZ+swCuydixY9WnTx/de++9WrZsmQ4dOqTnnntOvXr1UufOnbV//35t2bJFX375pWJjYyVJeXl5Gj16tO677z4dPXpUO3bs0D333CMe1WcdPJgRljVu3DidPXtWmzZtUvfu3dWhQwetWrVKmzZt0vDhw2UYhuLj43XmzBlt27bN3G769On617/+pcOHD3uwewBXU1BQoKioKH399dd67bXX9MEHH2jv3r3aunWrWXP69GlFRETo2LFjKi4uVkxMjD799FN+wcCimCECJM2fP19r1qzRkSNH3MaPHj2qnj17uo317NlTH330kSoqKmqzRQDXITQ0VA8++KDat2+v4cOHKzs7W9u3b1ejRo3M5ZZbbpEkffzxx+rUqZP69Omj6OhojRgxQitXruTeQoshEAGS7rzzTg0YMEB//vOf3cYNw6j0jTMmVYG6wdfXV76+3/1k58WLFzV06FDl5OS4LR999JHuvPNO+fj4KCMjQ2+88YY6dOigpUuXql27dsrNzfXwWaC2WObHXYGf8swzz6hz585q27atOdahQwft2rXLrS4rK0tt27aVj49PbbcIoIq6du2qDRs2qFWrVmZI+jGbzaaePXuqZ8+emjNnjlq2bKn09HRNmTKllruFJzBDBHwvOjpa8fHxWrp0qTmWlJSkt956S0899ZSOHz+uNWvWaNmyZZo6daoHOwVwvR5++GF98803Gj16tPbt26dPPvlE27Zt03333aeKigrt3btXycnJ2r9/v06ePKmNGzfqzJkzat++vadbRy0hEAE/8NRTT7ldEuvatateffVV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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data = data1 ,x='EverBenched',hue='LeaveOrNot')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4f13f0e0",
   "metadata": {},
   "source": [
    "## in this graph we seen that most of employees who are benched they didnt leave the company"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "5b7ffe84",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='ExperienceInCurrentDomain', ylabel='count'>"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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fftWq3Hs2LFauXKlcnJyXJfidu/erbfeektvv/22232wHh4eyszM1MCBA/XEE0+odevW1Y5XUFCg/v37KyMjQwMGDNCGDRs0cuRIBQUFaejQoVq5cqU6deqkxx57TI8++mhdf+TVEHYAAKhH8+bN04033qjMzExX2+uvv66IiAjt3LlT0dHRuv/++932WbBggUJCQrRt2zbFxMRowIABevLJJ7V+/Xp1795dkrR8+XINHDhQTZo00Z49e/Tmm2/q4MGDCg8PlySNGzdOq1ev1sKFC5WZmamdO3fq+uuvr7HG0+07d+50tVVWVmrp0qVq1apVtf733XefOnfurMmTJ2vBggXVts+cOVO9evXSn/70J0lSdHS0tm3bpunTp2vo0KEKDAyUh4eH/P395XA4LuXjrBUuYwEAUI8KCgr0ySefqFmzZq7luuuukyTXpao9e/Zo4MCBuvrqq9W8eXO1a9dOkrR//35JUqtWrdSnTx+98cYbkn59f+Q///lPDRo0SJK0efNmGYah6Ohot/Pk5eW5XQ67kDMvJ0VGRtYYdE6bOnWqFi9erG3btlXbtn37dnXr1s2trVu3btq1a5eqqqouup66wswOAAD16NSpU+rbt6+mTp1abdvpl1v27dtXERERevXVVxUeHq5Tp04pJiZGlZWVrr6DBg3SmDFjNGfOHC1fvlwdOnRQp06dXOfw8PBQQUFBtUevNGvWTNL/za7UZPv27ZKkqKgoV5ufn995x3Xrrbfq9ttv18SJEzV06FC3bTW9tun0PT9mIOwAAFCPbrzxRr399ttq27atPD2r/9k9dOiQtm/frldeecV1iWr9+vXV+t17770aPny4Vq9ereXLl2vw4MGubV26dFFVVZVKS0tdxzjbgw8+qIEDB+q9996rdt/OjBkzFBQUpD59+lzS2F588UV17txZ0dHRbu033HBDtTFs2LBB0dHRrjDm7e3dYLM8XMYCAKCOlJWVub0GqbCwUMOHD9dPP/2kP/zhD/rXv/6lb775RmvXrtXDDz+sqqoqtWzZUkFBQZo/f752796tjz/+WOnp6dWO7efnp3vuuUd/+tOftH37dtfbBKRfZ20GDRqkIUOGaOXKldq7d6/y8/M1depUffDBB5J+DTv33XefUlNTtWDBAu3bt09fffWVhg8frnfffVevvfbaBWdzzhYbG6tBgwZpzpw5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0Dg16pmds4WFhSkyMlK7du2SJDkcDlVWVurw4cNu/UpLSxUaGmpGiQAAoJG5rMLOoUOHdODAAYWFhUmS4uLi5OXlpZycHFef4uJibdmyRYmJiWaVCQAAGhFTL2MdO3ZMu3fvdq3v3btXhYWFCgwMVGBgoDIyMnT//fcrLCxM+/bt08SJExUcHKz77rtPkhQQEKBhw4Zp7NixCgoKUmBgoMaNG6fY2FjXt7MAAMCVzdSws2nTJt12222u9fT0dElSamqq5s2bp6KiIi1ZskRHjhxRWFiYbrvtNq1YsUL+/v6ufWbNmiVPT0/1799fFRUV6tWrlxYtWiQPD48GHw8AAGh8TA07SUlJMgzjnNvXrFlzwWP4+Phozpw5mjNnTl2WBgAALOKyumcHAADgUhF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApRF2AACApXmaXQCk/VNi6/0cbZ4tqvdzAADQGDGzAwAALI2wAwAALI2wAwAALI2wAwAALM3UsPPpp5+qb9++Cg8Pl81m0zvvvOO23TAMZWRkKDw8XL6+vkpKStLWrVvd+jidTo0ePVrBwcHy8/PT3XffrYMHDzbgKAAAQGNmatg5fvy4OnXqpLlz59a4fdq0aZo5c6bmzp2r/Px8ORwO9enTR0ePHnX1SUtL06pVq5Sdna3169fr2LFjuuuuu1RVVdVQwwAAAI2YqV89T0lJUUpKSo3bDMPQ7NmzNWnSJPXr10+StHjxYoWGhmr58uUaPny4ysrKtGDBAi1dulS9e/eWJC1btkwRERFat26dbr/99hqP7XQ65XQ6Xevl5eV1PDIAANBYNNp7dvbu3auSkhIlJye72ux2u3r06KENGzZIkgoKCnTy5Em3PuHh4YqJiXH1qUlWVpYCAgJcS0RERP0NBAAAmKrRhp2SkhJJUmhoqFt7aGioa1tJSYm8vb3VsmXLc/apyYQJE1RWVuZaDhw4UMfVAwCAxqLRP0HZZrO5rRuGUa3tbBfqY7fbZbfb66Q+AADQ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cV8JGDfjvhJcDuO2GcaFvq8FAABw+brs79kBAAA4H8IOAACwNMIOAACwNMIOAACwNMKOiV5++WW1a9dOPj4+iouL02effWZ2SfXu008/Vd++fRUeHi6bzaZ33nnH7JLqXVZWlm666Sb5+/srJCRE9957r3bs2GF2WQ1i3rx56tixo+thYwkJCfrwww/NLqtBZWVlyWazKS0tzexS6l1GRoZsNpvb4nA4zC6r3n333Xf6z//8TwUFBalp06bq3LmzCgoKzC6r3rVt27baz9tms2nUqFFml1YNYcckK1asUFpamiZNmqQvv/xS3bt3V0pKivbv3292afXq+PHj6tSpk+bOnWt2KQ0mLy9Po0aN0saNG5WTk6NffvlFycnJOn78uNml1bvWrVvrxRdf1KZNm7Rp0yb17NlT99xzj7Zu3Wp2aQ0iPz9f8+fPV8eOHc0upcF06NBBxcXFrqWoyNpv6D58+LC6desmLy8vffjhh9q2bZtmzJhh+lP5G0J+fr7bz/r0w30feOABkyurQZ28jROX7He/+50xYsQIt7brrrvOeOaZZ0yqqOFJMlatWmV2GQ2utLTUkGTk5eWZXYopWrZsabz22mtml1Hvjh49akRFRRk5OTlGjx49jDFjxphdUr2bPHmy0alTJ7PLaFBPP/20ccstt5hdRqMwZswY45prrjFOnTpldinVMLNjgsrKShUUFCg5OdmtPTk5WRs2bDCpKjSUsrIySVJgYKDJlTSsqqoqZWdn6/jx40pISDC7nHo3atQo/f73v1fv3r3NLqVB7dq1S+Hh4WrXrp0efPBBffPNN2aXVK/effddxcfH64EHHlBISIi6dOmiV1991eyyGlxlZaWWLVumhx9+uMFfqH0xCDsm+PHHH1VVVVXtRaWhoaHVXmgKazEMQ+np6brlllsUExNjdjkNoqioSM2aNZPdbteIESO0atUq3XDDDWaXVa+ys7O1efNmZWVlmV1Kg+ratauWLFmiNWvW6NVXX1VJSYkSExN16NAhs0urN998843mzZunqKgorVmzRiNGjNATTzyhJUuWmF1ag3rnnXd05MgRDR061OxSamSJ10Vcrs5Ov4ZhNMpEjLrz+OOP66uvvtL69evNLqXBtG/fXoWFhTpy5IjefvttpaamKi8vz7KB58CBAxozZozWrl0rHx8fs8tpUCkpKa5/x8bGKiEhQddcc40WL16s9PR0EyurP6dOnVJ8fLwyMzMlSV26dNHWrVs1b948DRkyxOTqGs6CBQuUkpKi8PBws0upETM7JggODpaHh0e1WZzS0tJqsz2wjtGjR+vdd9/VJ598otatW5tdToPx9vbWtddeq/j4eGVlZalTp0566aWXzC6r3hQUFKi0tFRxcXHy9PSUp6en8vLy9Je//EWenp6qqqoyu8QG4+fnp9jYWO3atcvsUupNWFhYteB+/fXXW/7LJmf69ttvtW7dOj3yyCNml3JOhB0TeHt7Ky4uznXn+mk5OTlKTEw0qSrUF8Mw9Pjjj2vlypX6+OOP1a5dO7NLMpVhGHI6nWaXUW969eqloqIiFRYWupb4+HgNGjRIhYWF8vDwMLvEBuN0OrV9+3aFhYWZXUq96datW7VHSezcufOyeBF1XVm4cKFCQkL0+9//3uxSzonLWCZJT0/X4MGDFR8fr4SEBM2fP1/79+/XiBEjzC6tXh07dky7d+92re/du1eFhYUKDAxUmzZtTKys/owaNUrLly/X3//+d/n7+7tm9AICAuTr62tydfVr4sSJSklJUUREhI4ePars7Gzl5uZq9erVZpdWb/z9/avdj+Xn56egoCDL36c1btw49e3bV23atFFpaamef/55lZeXKzU11ezS6s2TTz6pxMREZWZmqn///vrXv/6l+fPna/78+WaX1iBOnTqlhQsXKjU1VZ6ejThSmPtlsCvbX//6VyMyMtLw9vY2brzxxiviq8iffPKJIanakpqaanZp9aam8UoyFi5caHZp9e7hhx92/Y63atXK6NWrl7F27Vqzy2pwV8pXzwcMGGCEhYUZXl5eRnh4uNGvXz9j69atZpdV79577z0jJibGsNvtxnXXXWfMnz/f7JIazJo1awxJxo4dO8wu5bxshmEY5sQsAACA+sc9OwAAwNIIOwAAwNIIOwAAwNIIOwAAwNIIOwAAwNIIOwAAwNIIOwAAwNIIOwAAwNIIO8AVbOjQobr33nvNLgOXKZvNpnfeecfsMoALIuwA9Wzo0KGy2WzVljvuuMPs0vTSSy9p0aJFZpfhJjc3VzabTUeOHLmk/crLyzVp0iRdd9118vHxkcPhUO/evbVy5Uo15gfFt23bVrNnz3ZrO/0Z2Gw2NWnSRAEBAerSpYvGjx+v4uJicwqtQXFxsVJSUswuA7igRvzWLsA67rjjDi1cuNCtzW63m1SNVFVVJZvNpoCAANNqqEtHjhzRLbfcorKyMj3//PO66aab5Onpqby8PI0fP149e/ZUixYtanXskydPysvLy62tsrJS3t7edVD5+e3YsUPNmzdXeXm5Nm/erGnTpmnBggXKzc1VbGxsvZ//QhwOh9klABeFmR2gAdjtdjkcDrelZcuWys3Nlbe3tz777DNX3xkzZig4ONj1f/BJSUl6/PHH9fjjj6tFixYKCgrS//t//89ttqKyslLjx4/XVVddJT8/P3Xt2lW5ubmu7YsWLVKLFi30j3/8QzfccIPsdru+/fbbapexDMPQtGnTdPXVV8vX11edOnXS3/72N9f20zMOH330keLj49W0aVMlJiZqx44dbuN99913FR8fLx8fHwUHB6tfv34XXevZTte+Zs0aXX/99WrWrJnuuOMOtxmOiRMnat++ffriiy+UmpqqG264QdHR0Xr00UdVWFioZs2aSar5skuLFi1cs1v79u2TzWbTW2+9paSkJPn4+GjZsmWuzykrK0vh4eGKjo6WJH333XcaMGCAWrZsqaCgIN1zzz3at2+f69in9/uv//ovhYWFKSgoSKNGjdLJkyddP9tvv/1WTz75pGsm50whISFyOByKjo7Wgw8+qM8//1ytWrXSH//4R1efU6dOacqUKWrdurXsdrs6d+7s9lb5M8fUvXt3+fr66qabbtLOnTuVn5+v+Ph412f6ww8/uPbLz89Xnz59FBwcrICAAPXo0UObN292q+/Mz/P0eVauXKnbbrtNTZs2VadOnfTPf/7znD9boMGY+hpS4AqQmppq3HPPPefc/tRTTxmRkZHGkSNHjMLCQsNutxsrV650be/Ro4fRrFkzY8yYMcbXX39tLFu2zGjatKnbm5UHDhxoJCYmGp9++qmxe/duY/r06Ybdbjd27txpGIZhLFy40PDy8jISExONzz//3Pj666+NY8eOVatt4sSJxnXXXWesXr3a2LNnj7Fw4ULDbrcbubm5hmH831vru3btauTm5hpbt241unfvbiQmJrqO8Y9//MPw8PAwnn32WWPbtm1GYWGh8cILL1x0rafPcfjwYbfae/fubeTn5xsFBQXG9ddfbwwcONAwDMOoqqoyWrZsaTz22GMX/FlIMlatWuXWFhAQ4HoD/d69ew1JRtu2bY23337b+Oabb4zvvvvOSE1NNZo1a2YMHjzY2LJli1FUVGQcP37ciIqKMh5++GHjq6++MrZt22YMHDjQaN++veF0Ol0/++bNmxsjRowwtm/fbrz33ntuP7tDhw4ZrVu3NqZMmWIUFxcbxcXFNX4GZ5o1a5Yhyfj+++8NwzCMmTNnGs2bNzfefPNN4+uvvzbGjx9veHl5uT7P02M6/XPdtm2bcfPNNxs33nijkZSUZKxfv97YvHmzce211xojRoxwneejjz4yli5damzbts3Ytm2bMWzYMCM0NNQoLy+v8fM88zz/+Mc/jB07dhj/8R//YURGRhonT5684M8GqE+EHaCepaamGh4eHoafn5/bMmXKFMMwDMPpdBpdunQx+vfvb3To0MF45JFH3Pbv0aOHcf311xunTp1ytT399NPG9ddfbxiGYezevduw2WzGd99957Zfr169jAkTJhiG8WtgkGQUFhZWq+102Dl27Jjh4+NjbNiwwa3PsGHDjD/84Q+GYfzfH+F169a5tr///vuGJKOiosIwDMNISEgwBg0aVONncTG11hR2JBm7d+929f/rX/9qhIaGGoZhGN9//70hyZg5c2aN5zzTxYad2bNnu/VJTU01QkNDXSHGMAxjwYIFRvv27d1+Lk6n0/D19TXWrFnj2i8yMtL45ZdfXH0eeOABY8CAAa71yMhIY9asWW7nO1/Y+fDDDw1JxhdffGEYhmGEh4e7hUnDMIybbrrJGDlypNuYXnvtNdf2N99805BkfPTRR662rKwso3379tXOd9ovv/xi+Pv7G++9956rraawc+Z5tm7dakgytm/ffs7jAg2Be3aABnDbbbdp3rx5bm2BgYGSJG9vby1btkwdO3ZUZGRktZtVJenmm292u8SRkJCgGTNmqKqqSps3b5ZhGK5LK6c5nU4FBQW51r29vdWxY8dz1rht2zb9/PPP6tOnj1t7ZWWlunTp4tZ25nHCwsIkSaWlpWrTpo0KCwv16KOP1niOi631bE2bNtU111zjds7S0lJJcl3OO/sS0G8RHx9frS02NtbtPp2CggLt3r1b/v7+bv1+/vln7dmzx7XeoUMHeXh4uNVeVFRU69rOHG95ebn+/e9/q1u3bm59unXrpv/93/91azvzZxYaGuoa05ltpz9T6def57PPPquPP/5Y33//vaqqqnTixAnt37//vPWd63fjuuuuu5RhAnWKsAM0AD8/P1177bXn3L5hwwZJ0k8//aSffvpJfn5+F33sU6dOycPDQwUFBW5/VCW57lWRJF9f3/MGglOnTkmS3n//fV111VVu286+mfrMG3ZPH/P0/r6+vr+51rOdfYOwzWZz/dFv1aqVWrZsqe3bt59z/5r2O+30/TNnqunzP7vt1KlTiouL0xtvvFGtb6tWrc5b++nPqjZOj7Nt27ZuxzyTYRjV2mr6mZ3ddmZdQ4cO1Q8//KDZs2crMjJSdrtdCQkJqqysPG995/vdAMxC2AFMtmfPHj355JN69dVX9dZbb2nIkCH66KOP1KTJ/31/YOPGjW77bNy4UVFRUfLw8FCXLl1UVVWl0tJSde/evdZ1nL5xef/+/erRo0etj9OxY0d99NFHeuihh6ptq6taz9SkSRMNGDBAS5cu1eTJkxUeHu62/fjx47Lb7fL09FSrVq3cbmzetWuXTpw4Uavz3njjjVqxYoVCQkLUvHnzWtfv7e2tqqqqi+pbUVGh+fPn69Zbb3UFqvDwcK1fv1633nqrq9+GDRv0u9/9rtY1SdJnn32ml19+WXfeeack6cCBA/rxxx9/0zEBs/BtLKABOJ1OlZSUuC0//vijqqqqNHjwYCUnJ+uhhx7SwoULtWXLFs2YMcNt/wMHDig9PV07duzQm2++qTlz5mjMmDGSpOjoaA0aNEhDhgzRypUrtXfvXuXn52vq1Kn64IMPLrpGf39/jRs3Tk8++aQWL16sPXv26Msvv9Rf//pXLV68+KKPM3nyZL355puaPHmytm/frqKiIk2bNq1Oaz1bZmamIiIi1LVrVy1ZskTbtm3Trl279Prrr6tz5846duyYJKlnz56aO3euNm/erE2bNmnEiBHVZl4u1qBBgxQcHKx77rlHn332mfbu3au8vDyNGTNGBw8evOjjtG3bVp9++qm+++67amGitLRUJSUl2rVrl7Kzs9WtWzf9+OOPbpdEn3rqKU2dOlUrVqzQjh079Mwzz6iwsND1+1Fb1157rZYuXart27friy++0KBBg847awc0ZszsAA1g9erVrvsXTmvfvr0GDhyoffv26b333pP063NLXnvtNfXv3199+vRR586dJUlDhgxRRUWFfve738nDw0OjR4/WY4895jrWwoUL9fzzz2vs2LH67rvvFBQUpISEBNf/lV+s5557TiEhIcrKytI333yjFi1a6MYbb9TEiRMv+hhJSUn6n//5Hz333HN68cUX1bx5c7dZh7qq9UwtW7bUxo0b9eKLL+r555/Xt99+q5YtWyo2NlbTp093PU9oxowZeuihh3TrrbcqPDxcL730kgoKCmp1zqZNm+rTTz/V008/rX79+uno0aO66qqr1KtXr0ua6ZkyZYqGDx+ua665Rk6n0+0yW/v27WWz2dSsWTNdffXVSk5OVnp6utvzbZ544gmVl5dr7NixKi0t1Q033KB3331XUVFRtRrXaa+//roee+wxdenSRW3atFFmZqbGjRv3m44JmMVmnH0BG0CjkpSUpM6dO9d44zIA4MK4jAUAACyNsAMAACyNy1gAAMDSmNkBAACWRtgBAACWRtgBAACWRtgBAACWRtgBAACWRtgBAACWRtgBAACWRtgBAACW9v8Bafqo3w4JJbAAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data = data1 ,x='ExperienceInCurrentDomain',hue='LeaveOrNot')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "da096d1a",
   "metadata": {},
   "source": [
    "from this grap we can know that employee not leaving or staying in company because of experience"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "1e1c5483",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Gender', ylabel='count'>"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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WJh69ep1Udf20ksvqV27dmrTpo3b+HXXXVflGrKystSmTRs7sDVs2LDOZou4ZQYAQB0qLi5WTk6O22PkyJE6fPiwHnroIW3cuFHfffedVq1apeHDh6uiokJNmjRRWFiY5s+fr927d+vjjz9WSkpKlXMHBgbq7rvv1sSJE5Wbm6vExER7X5s2bTRo0CANGTJEy5YtU15enjZt2qSpU6dqxYoVkn4ORPfee6+GDh2qBQsWaO/evfr66681cuRIvf/++/rzn//8q7NCvxQfH69BgwZp9uzZbuOjR4/W2rVr9cILL2jXrl1avHix5syZozFjxtg1MTEx+uSTT/TDDz+4feKuNhCIAACoQ+vXr1f79u3dHpMmTdJnn32miooK9enTR3FxcXrqqacUEhKiBg0aqEGDBsrIyFB2drbi4uL09NNPa9q0aWc9/6BBg/TVV1+pW7duat68udu+hQsXasiQIRo9erSuueYaDRgwQP/4xz8UHR0t6ef1Qe+8844mTJigmTNn6tprr1W3bt30/fffa926ddX+UskXXnihyu2wDh066J133lFGRobi4uI0adIkTZkyxe3W2pQpU7R3715dffXV512rVBMclidv2NUjJSUlCgkJUXFxsYKDg2vtefjYvffgY/cALtbJkyeVl5enli1byt/f39PtGON8r/uF/v1mhggAABiPQAQAAIxHIAIAAMYjEAEAAOMRiAAAgPEIRAAAwHgeDUSffPKJ+vfvr6ioKDkcDr333ntu+y3LUmpqqqKiohQQEKCEhARt377draasrEyjRo1Ss2bNFBgYqAEDBujAgQNuNUVFRRo8eLBCQkIUEhKiwYMH68iRI7V8dQAAoL7waCA6fvy4brzxRs2ZM+es+19++WXNmDFDc+bM0aZNm+RyudSrVy8dPXrUrklOTtby5cuVkZGhDRs26NixY+rXr5/bV30nJiYqJydHK1eu1MqVK5WTk+P2deYAAMBsHv0ts759+6pv375n3WdZlmbNmqUJEybovvvukyQtXrxYERERWrp0qUaOHKni4mItWLBAb775pnr27ClJWrJkiaKjo7VmzRr16dNHubm5Wrlypb744gt17txZkvTaa6+pS5cu2rlzp6655pq6uVgAAOC1vPbHXfPy8lRQUKDevXvbY06nU927d1dWVpZGjhyp7OxsnTp1yq0mKipKcXFxysrK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a+3atVZwcLB18uRJt2Ouvvpqa968eRd9vfA8Y366A2Zr1qyZ7rrrLi1evNj+P7tmzZq51ezZs0cTJ07UF198oZ9++smeGdq3b5/i4uKqnDM7O1vr1q1T48aNq+zbs2eP2rRpUzsXAxjitttu09y5c+3twMBAtW7dWps2bXKbEaqoqNDJkyd14sQJNWrUSJJ0ww032PsjIiIkSfHx8W5jJ0+eVElJiYKDg3X8+HFNnjxZH374oX788UedPn1apaWl55whys7O1rFjxxQWFuY2Xlpa+qu32uGdCEQwxvDhw+0p8v/5n/+psr9///6Kjo7Wa6+9pqioKFVWViouLk7l5eVnPV9lZaX69++vqVOnVtkXGRlZs80DBjoTgP5dZWWlJk+erPvuu69Kvb+/v/1vPz8/+98Oh+OcY2f+x2fs2LH66KOP9Kc//UmtW7dWQECAfve73533/R8ZGan169dX2RcaGnphFwivQiCCMe644w77P259+vRx23fo0CHl5uZq3rx56tatmyRpw4YN5z1fhw4d9O677yomJka+vryVgLrQoUMH7dy5s0pQulSffvqphg0bpnvvvVfSz2uK9u7de94+CgoK5Ovrq5iYmBrtBZ7BomoYw8fHR7m5ucrNzZWPj4/bviZNmigsLEzz58/X7t279fHHHyslJeW853v88cd1+PBhPfTQQ9q4caO+++47rVq1SsOHD1dFRUVtXgpgrEmTJumNN95Qamqqtm/frtzcXL399tt67rnnLum8rVu31rJly5STk6OvvvpKiYmJ5/xAhST17NlTXbp00T333KOPPvpIe/fuVVZWlp577jl70TbqFwIRjBIcHKzg4OAq4w0aNFBGRoays7MVFxenp59+WtOmTTvvuaKiovTZZ5+poqJCffr0UVxcnJ566imFhISoQQPeWkBt6NOnjz788EOtXr1aN910k37zm99oxowZatGixSWdd+bMmWrSpIm6du2q/v37q0+fPurQocM56x0Oh1asWKFbb71Vw4cPV5s2bfTggw9q79699pol1C8Oy7IsTzcBAADgSfxvLAAAMB6BCAAAGI9ABAAAjEcgAgAAxiMQAQAA4xGIAACA8QhEAADAeAQiAABgPAIRAFyAhIQEJScne7oNALWEQASg3igoKNBTTz2l1q1by9/fXxEREbrlllv06quv6sSJE55uD0A9xk90A6gXvvvuO918880KDQ1VWlqa4uPjdfr0ae3atUuvv/66oqKiNGDAAE+3eU4VFRVyOBz8zh3gpXhnAqgXkpKS5Ovrq82bN2vgwIFq27at4uPj9dvf/lZ///vf1b9/f0lScXGxHnvsMYWHhys4OFi33367vvrqK/s8qampateund58803FxMQoJCREDz74oI4ePWrXHD9+XEOGDFHjxo0VGRmp6dOnV+mnvLxc48aN05VXXqnAwEB17txZ69evt/cvWrRIoaGh+vDDD3XdddfJ6XTq+++/r70XCMAlIRAB8HqHDh3SqlWr9PjjjyswMPCsNQ6HQ5Zl6a677lJBQYFWrFih7OxsdejQQT169NDhw4ft2j179ui9997Thx9+qA8//FCZmZl66aWX7P1jx47VunXrtHz5cq1atUrr169Xdna22/M9/PDD+uyzz5SRkaGvv/5a999/v+644w59++23ds2JEyeUnp6uP//5z9q+fbvCw8Nr+JUBUGMsAPByX3zxhSXJWrZsmdt4WFiYFRgYaAUGBlrjxo2z1q5dawUHB1snT550q7v66qutefPmWZZlWc8//7zVqFEjq6SkxN4/duxYq3PnzpZlWdbRo0ethg0bWhkZGfb+Q4cOWQEBAdZTTz1lWZZl7d6923I4HNYPP/zg9jw9evSwxo8fb1mWZS1cuNCSZOXk5NTMiwCgVrGGCEC94XA43LY3btyoyspKDRo0SGVlZcrOztaxY8cUFhbmVldaWqo9e/bY2zExMQoKCrK3IyMjVVhYKOnn2aPy8nJ16dLF3t+0aVNdc8019vaXX34py7LUpk0bt+cpKytze+6GDRvqhhtuuIQrBlBXCEQAvF7r1q3lcDj0zTffuI23atVKkhQQECBJqqysVGRkpNtanjNCQ0Ptf/v5+bntczgcqqyslCRZlvWr/VRWVsrHx0fZ2dny8fFx29e4cWP73wEBAVVCHADvRCAC4PXCwsLUq1cvzZkzR6NGjTrnOqIOHTqooKBAvr6+iomJqdZztW7dWn5+fvriiy/UvHlzSVJRUZF27dql7t27S5Lat2+viooKFRYWqlu3btV6HgDehUXVAOqFV155RadPn1anTp309ttvKzc3Vzt37tSSJUv0zTffyMfHRz179lSXLl10zz336KOPPtLevXuVlZWl5557Tps3b76g52ncuLFGjBihsWPHau3atdq2bZuGDRvm9nH5Nm3aaNCgQRoyZIiWLVumvLw8bdq0SVOnTtWKFStq6yUAUIuYIQJQL1x99dXasmWL0tLSNH78eB04cEBOp1PXXXedxowZo6SkJDkcDq1YsUITJkzQ8OHDdfDgQblcLt16662KiIi44OeaNm2ajh07pgEDBigoKEijR49WcXGxW83ChQv1X//1Xxo9erR++OEHhYWFqUuXLrrzzjtr+tIB1AGHdSE3zAEAAC5j3DIDAADGIxABAADjEYgAAIDxCEQAAMB4BCIAAGA8AhEAADAegQgAABiPQAQAAIxHIAIAAMYjEAEAAOMRiAAAgPH+D2pzTuJ59VznAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data = data1 ,x='Gender',hue='LeaveOrNot')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "23ba02e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "## almost equal laving and staying of employees are female , most of mens are not leaving the company"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "5a94c640",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\91772\\AppData\\Local\\Temp\\ipykernel_10040\\1308909575.py:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  data1['PaymentTier'] = data1['PaymentTier'].astype('category')\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='PaymentTier', ylabel='count'>"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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ffVVTZ8+XXv37jXHwsLC1KpVK7Vq1eqi3qe2tlZBQUEXtQ8A9TFDBOCSCgkJkdPplNvtVmpqqtLS0rR27VqtWrVKPXr0UEREhJxOp1JTU1VWViZJMgxDV199tf7nf/7Ha187d+5UixYt9OWXX0o6NSuzePFiDR48WC1btlSXLl30wQcf6IsvvlBycrLCw8OVlJRk1p/2pz/9SYmJiQoNDVWnTp00Y8YMnThxwlxvs9n04osv6p577lHLli0VFxenN998U5L01VdfmT9S3KZNG3P2x+l0mi+73S6bzVZv7EynzJYvX64uXbooNDRU1157rZ5//nlz3VdffSWbzabXXntNycnJCg0N1apVqxrnPwwALwQiAE0qLCxMtbW1qqmp0dNPP61PP/1Ua9euVXFxsfkr8zabTQ888ICWL1/ute2yZcvUu3dv/eQnPzHHnn76aY0cOVJFRUW69tprlZqaqrFjx2rq1Knavn27JOnRRx816//85z/rF7/4hSZMmKDdu3dr8eLFys3N1TPPPOP1XjNmzFBKSoo+++wz3XnnnUpLS9N3330nt9ut119/XZK0d+9elZSU6Le//W2DPosXXnhB06ZN0zPPPKM9e/YoOztbTz75pFasWOFVN2XKFE2YMEF79uzRwIEDG/ReAH6EAQCXyKhRo4y7777bXP7www+N6OhoIyUlpV7tRx99ZEgyjhw5YhiGYXzzzTdGQECA8eGHHxqGYRg1NTVGu3btjNzcXHMbScavf/1rc/mDDz4wJBlLly41x1555RUjNDTUXO7du7eRnZ3t9d4rV6402rdvf9b9Hj161LDZbMY777xjGIZhbNiwwZBklJeXn/G4ly9fbtjt9nrjTz31lNGtWzdz2e12G6tXr/aqefrpp42kpCTDMAyjuLjYkGTMnz//jO8DoPFwDRGAS+qtt95Sq1atdOLECdXW1uruu+/WggUL9MknnygrK0tFRUX67rvvdPLkSUnS/v371bVrV7Vv31533XWXli1bpp/+9Kd66623dPz4cd13331e+7/++uvNfzscDklSQkKC19jx48dVUVGhyMhIFRYWatu2bV4zQnV1dTp+/LiOHTumli1b1ttveHi4IiIizFN6jeHQoUM6cOCA0tPT9eCDD5rjJ06ckN1u96rt0aNHo70vgDMjEAG4pG677TYtWrRIQUFBcrlcCgoKUmVlpQYMGKABAwZo1apVateunfbv36+BAweqpqbG3HbMmDEaMWKE5s2bp+XLl2v48OFmYDnt+xcY22y2s46dDlwnT57UjBkzdO+999brNTQ09Iz7Pb2f0/toDKf39cILL6hnz55e6wICAryWw8PDG+19AZwZgQjAJRUeHq6rr77aa+zzzz/Xt99+q2effVZut1uSzOt9vu/OO+9UeHi4Fi1apHfeeUebNm266H5uvPFG7d27t15PFyI4OFjSqZmlhnI4HLriiiv097//XWlpaQ3eD4DGQSAC0OQ6dOig4OBgLViwQA8//LB27typp59+ul5dQECARo8eralTp+rqq69WUlLSRb/39OnTNXjwYLndbt13331q0aKFPvvsM+3YsUO/+c1vzmsfHTt2lM1m01tvvaU777zTvKX+QmVlZWnChAmKjIzUoEGDVF1dre3bt6u8vFyZmZkXvD8ADcddZgCaXLt27ZSbm6vf//736tq1q5599tl6t9iflp6erpqaGj3wwAON8t4DBw7UW2+9pfz8fN1000362c9+prlz56pjx47nvY8rrrhCM2bM0BNPPCGHw+F1F9uFGDNmjF588UXl5uYqISFBffr0UW5urmJjYxu0PwANZzMMw/B1EwBwNu+//76Sk5N18OBB86JpAGhsBCIAfqm6uloHDhzQQw89pPbt2+vll1/2dUsAmjFOmQHwS6+88oquueYaeTwezZ4929ftAGjmmCECAACWxwwRAACwPAIRAACwPAIRAACwPAIRAACwPAIRAACwPAIRAACwPAIRAACwPAIRAACwvP8HQxl/sUd/fPUAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "data1['PaymentTier'] = data1['PaymentTier'].astype('category')\n",
    "sns.countplot(data = data1 ,x='PaymentTier',hue='LeaveOrNot')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d2d19f8d",
   "metadata": {},
   "source": [
    "## in this half of the emoployees having payment tier 3  are staying  and leaving \n",
    "\n",
    "## paytier 2 has high leaving of employees from company"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "e6b470b4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='City', ylabel='count'>"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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4zVhwgRNmzZNTZs2lcPhUEpKits2xcXF+uUvf6kWLVooJCRE9957r/75z39e+4nhuuvdu7ccDofS0tIuOe5Sny0ZGRmKiYkxx54/M/TCCy+YbQkJCZoxY8YlX6Nx48ZyOBxq06aN+vbtq9dff13Dhg3T448/7hbGavM5d/LkSd1///0KDAxUVFSUNmzYYPZxGQuWUVZWpldeeUW33HKLwsLCJEnBwcFauXKl9u/fr9///vdatmyZFi1a5Lbd4cOHtX79ev31r3/VX//6V+Xk5Lid8p08ebLef/99bdiwQdnZ2Xrvvff08ccfu+2jsrJSzz77rP75z39q/fr1ys/P14gRIy5aa15enhITEzV06FDt2bNHKSkpmjVrllauXOk2bv78+YqOjlZeXp5mzZqlPXv2KCEhQUOGDNHu3bv16quvavv27Xr88cev7s2D1wsICHAL6pezatUqBQUF6cMPP9S8efM0e/Zs82dmDMPQz3/+c7lcLr311lvKy8tT586d1atXL3399dfXagrwEB8fH6WmpiojI0MnTpy44JjLfbbEx8dr3759OnnypCQpJydHzZo1U05OjiTp22+/VW5uruLi4q64vkmTJqm0tNQ8Pmv7OffMM88oMTFRu3fv1n333adhw4bduMezAcsYPny44ePjYwQFBRlBQUGGJKNly5ZGXl7eRbeZN2+eERsba64//fTTRmBgoFFSUmK2PfHEE0aXLl0MwzCMkpISw8/Pz/jTn/5k9p86dcoIDAw0Jk6ceNHX2bFjhyHJKC0tNQzDMLZu3WpIMoqKigzDMIykpCSjT58+bts88cQTxu23326ut27d2hg8eLDbmIceesj45S9/6db23nvvGQ0aNDDKy8svWg/ql+HDhxuDBg0y1z/88EMjLCzMSExMrNZnGIYxceJEIy4uzlyPi4sz7r77brcxd911lzF9+nTDMAxjy5YtRkhIiPHNN9+4jbn55puNl156qU7nAs/6/vHy05/+1Bg5cqRhGIaRlZVlfP9P4uU+W86dO2c0a9bMeP311w3DMIyOHTsaaWlpRosWLQzDMIzc3FzD19fX/My7EElGVlZWtfby8nJDkjF37twa1WIY330+Llq0yG3fv/nNb8z1srIyw2azGX//+98Nw6j+GWx1nNmxmHvuuUe7du3Srl279OGHH6pv377q16+fvvjiC0nS66+/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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data = data1 ,x='City',hue='LeaveOrNot')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a288e0da",
   "metadata": {},
   "source": [
    "## major bamgalore and delhi employees not leaving much "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "1fbc2ad7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Education', ylabel='count'>"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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35577jm1bdtWzZs3dxq//fbbq+3Drl271Lx5c0dg8/Pzu26zRW49ZPbhhx+qX79+stlsslgsTpfzlZeXa/r06WrVqpWCg4Nls9k0YsQIfffdd07bKC0t1YQJE1S3bl0FBwerf//+OnHihFNNQUGBhg8fLqvVKqvVquHDh+vUqVPXYQ8BAHBWWFiorKwsp8e4ceP0ww8/6Le//a327Nmjf/3rX9qyZYtGjx6tiooK1alTR+Hh4Vq1apUOHTqkf/zjH0pKSqq27eDgYA0YMECzZ89Wdna2EhISHMuaN2+uYcOGacSIEdq0aZNycnKUkZGh+fPn691335V0IRANGjRII0eO1Jo1a3TkyBF9/vnnGjdunN566y397//+78/OCv1Uq1atNGzYMC1btsxpfPLkydq2bZv++Mc/6uDBg1q/fr2WL1+uKVOmOGoaNWqkDz/8UN9++63TFXfXglsD0ZkzZ9SmTRstX7682rKzZ89q3759mj17tvbt26dNmzbp4MGD6t+/v1NdYmKi0tLSlJqaqp07d6q4uFjx8fFOiTIhIUFZWVnavHmzNm/erKysLKfUDADA9fLBBx/ojjvucHrMmTNHH330kSoqKtS7d2/FxMToySeflNVqlZeXl7y8vJSamqrMzEzFxMRo0qRJWrhw4UW3P2zYMH322Wfq0qWLGjRo4LRs7dq1GjFihCZPnqxbb71V/fv31yeffCK73S7pwvlBr732mmbNmqUlS5botttuU5cuXXT06FFt3769xl8q+cc//rHa4bB27drptddeU2pqqmJiYjRnzhzNmzfP6dDavHnzdOTIEd1yyy2XPVfJFSyGOw/Y/YjFYlFaWtpl3+yMjAzdddddOnr0qBo0aKDCwkLVq1dPGzZs0NChQyVJ3333nex2u95991317t1b2dnZuv3227V792516NBBkrR792517NhRX331lW699dYr6q+oqEhWq1WFhYUKDQ2t0T5y2f0FXHYP4GZ17tw55eTkqHHjxgoICHB3O6Zxuff9Sv9+31BXmRUWFspisah27dqSLpyVX15erl69ejlqbDabYmJitGvXLknSxx9/LKvV6ghDknT33XfLarU6ai6mtLRURUVFTg8AAHBzumEC0blz5zRjxgwlJCQ4El5eXp78/PxUp04dp9rIyEjl5eU5aiIiIqptLyIiwlFzMSkpKY5zjqxW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FRWl9PR0x1hZWZl27NihTp06SZJiY2Pl6+vrVJObm6svvvjCUdOxY0cVFhZqz549jppPPvlEhYWFjhoAAGBubr2XWXFxsQ4dOuR4npOTo6ysLIWFhalBgwZKTExUcnKymjVrpmbNmik5OVlBQUFKSEiQJFmtVo0ZM0aTJ09WeHi4wsLCNGXKFLVq1Uo9evSQJLVo0UJ9+vTR2LFjtXLlSknSo48+qvj4+Cu+wgwAANzc3BqI9u7dq27dujmeJyUlSZJGjhypdevWadq0aSopKdH48eNVUFCgDh06aMuWLQoJCXGss2TJEvn4+GjIkCEqKSlR9+7dtW7dOqcb6r388suaOHGi42q0/v37X/K7jwAAgPl4zPcQeTq+h8h1+B4iAMD1csN/DxEAAMD1QiACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACm59GB6Pz58/qv//ovNW7cWIGBgWrSpInmzZunyspKR41hGJo7d65sNpsCAwMVFxenL7/80mk7paWlmjBhgurWravg4GD1799fJ06cuN67AwAAPJRHB6L58+frhRde0PLly5Wdna0FCxZo4cKFWrZsmaNmwYIFWrx4sZYvX66MjAxFRUWpZ8+eOn36tKMmMTFRaWlpSk1N1c6dO1VcXKz4+HhVVFS4Y7cAAICH8XF3A5fz8ccfa8CAAbr//vslSY0aNdIrr7yivXv3SrowO7R06VLNmjVLgwcPliStX79ekZGR2rhxo8aNG6fCwkKtWbNGGzZsUI8ePSRJL730kux2u7Zu3arevXtf9LVLS0tVWlrqeF5UVHQtdxUAALiRR88Qde7cWdu2bdPBgwclSZ999pl27typX//615KknJwc5eXlqVev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Vq9e7Viek5OjvLw89erVyzHm7++vrl27ateuXZKkzMxMlZeXO9XYbDbFxMQ4ai4mJSVFVqvV8bDb7S7bLwAA4FlqNEOUkpKiyMhIjR071mn8xRdf1MmTJzV9+nSXNPevf/1LK1asUFJSkv7whz9oz549mjhxovz9/TVixAjl5eVJUrXzmSIjI3X06FFJUl5envz8/FSnTp1qNVXrX8zMmTOdblFSVFREKAIA4CZVoxmilStX6rbbbqs23rJlS73wwgu/uKkqlZWVateunZKTk3XHHXdo3LhxGjt2bLWr2CwWi9NzwzCqjf3Uz9X4+/srNDTU6QEAAG5ONQpEeXl5io6OrjZer1495ebm/uKmqkRHR+v22293GmvRooWOHTsmSYqKinL082P5+fmOWaOoqCiVlZWpoKDgkjUAAMDcahSI7Ha7Pvroo2rjH330kWw22y9uqso999yjr7/+2mns4MGDatiwoSSpcePGioqKUnp6umN5WVmZduzYoU6dOkmSYmNj5evr61STm5urL774wlEDAADMrUbnED3yyCNKTExUeXm57rvvPknStm3bNG3aNJd+U/WkSZPUqVMnJScna8iQIdqzZ49WrVqlVatWSbpwqCwxMVHJyclq1qyZmjVrpuTkZAUFBSkhIUGSZLVaNWbMGE2ePFnh4eEKCwvTlClT1KpVK8dVZwAAwNxqFIimTZumH374QePHj1dZWZkkKSAgQNOnT9fMmTNd1tydd96ptLQ0zZw5U/PmzVPjxo21dOlSDRs2zKmXkpISjR8/XgUFBerQoYO2bNmikJAQR82SJUvk4+OjIUOGqKSkRN27d9e6devk7e3tsl4BAMCNy2IYhlHTlYuLi5Wdna3AwEA1a9ZM/v7+ruzNoxQVFclqtaqwsLDGJ1jHTv2Li7u6MaWFLHR3Cx6jwZz97m4BAG5qV/r3u0YzRFVq1aqlO++885dsAgAAwO1qdFI1AADAzYRABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATI9ABAAATO+GCkQpKSmyWCxKTEx0jBmGoblz58pmsykwMFBxcXH68ssvndYrLS3VhAkTVLduXQUHB6t///46ceLEde4eAAB4qhsmEGVkZGjVqlVq3bq10/iCBQu0ePFiLV++XBkZGYqKilLPnj11+vRpR01iYqLS0tKUmpqqnTt3qri4WPHx8aqoqLjeuwEAADzQDRGIiouLNWzYMK1evVp16tRxjBuGoaVLl2rWrFkaPHiwYmJitH79ep09e1YbN26UJBUWFmrNmjVatGiRevTooTvuuEMvvfSS9u/fr61bt17yNUtLS1VUVOT0AAAAN6cbIhA9/vjjuv/++9WjRw+n8ZycHOXl5alXr16OMX9/f3Xt2lW7du2SJGVmZqq8vNypxmazKSYmxlFzMSkpKbJarY6H3W538V4BAABP4fGBKDU1Vfv27VNKSkq1ZXl5eZKkyMhIp/HIyEjHsry8PPn5+TnNLP205mJmzpypwsJCx+P48eO/dFcAAICH8nF3A5dz/PhxPfnkk9qyZYsCAgIuWWexWJyeG4ZRbeynfq7G399f/v7+V9cwAAC4IXn0DFFmZqby8/MVGxsrHx8f+fj4aMeOHfrzn/8sHx8fx8zQT2d68vPzHcuioqJUVlamgoKCS9YAAABz8+hA1L17d+3fv19ZWVmOR/v27TVs2DBlZWWpSZMmioqKUnp6umOdsrIy7dixQ506dZIkxcbGytfX16kmNzdXX3zxhaMGAACYm0cfMgsJCVFMTIzTWHBwsMLDwx3jiYmJSk5OVrNmzdSsWTMlJycrKChICQkJkiSr1aoxY8Zo8uTJCg8PV1hYmKZMmaJWrVpVO0kbAACYk0cHoisxbdo0lZSUaPz48SooKFCHDh20ZcsWhYSEOGqWLFkiHx8fDRkyRCUlJerevbvWrVsnb29vN3YOAAA8hcUwDMPdTdwIioqKZLVaVVhYqNDQ0BptI3bqX1zc1Y0pLWShu1vwGA3m7Hd3CwBwU7vSv98efQ4RAADA9UAgAgAApkcgAgAApkcgAgAApkcgAgAApkcgAgAApkcgAgAApkcgAgAApkcgAgAApkcgAgAApkcgAgAApkcgAgAApkcgAgAApufj7gYAuFfs1L+4uwWPkLlwhLtbAOBGzBABAADTIxABAADTIxABAADTIxABAADT46RqAJB0bF4rd7fgERrM2e/uFgC3YIYIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYHoEIAACYnkcHopSUFN15550KCQlRRESEBg4cqK+//tqpxjAMzZ07VzabTYGBgYqLi9OXX37pVFNaWqoJEyaobt26Cg4OVv/+/XXixInruSsAAMCDeXQg2rFjhx5//HHt3r1b6enpOn/+vHr16qUzZ844ahYsWKDFixdr+fLlysjIUFRUlHr27KnTp087ahITE5WWlqbU1FTt3LlTxcXFio+PV0VFhTt2CwAAeBgfdzdwOZs3b3Z6vnbtWkVERCgzM1P33nuvDMPQ0qVLNWvWLA0ePFiStH79ekVGRmrjxo0aN26cCgsLtWbNGm3YsEE9evSQJL300kuy2+3aunWrevfufd33CwAAeBaPniH6qcLCQklSWFiYJCknJ0d5eXnq1auXo8bf319du3bVrl27JEmZmZkqLy93qrHZbIqJiXHUXExpaamKioqcHgAA4OZ0wwQiwzCUlJSkzp07KyYmRpKUl5cnSYqMjHSqjYyMdCzLy8uTn5+f6tSpc8mai0lJSZHVanU87Ha7K3cHAAB4kBsmED3xxBP6/PPP9corr1RbZrFYnJ4bhlFt7Kd+rmbmzJkqLCx0PI4fP16zxgEAgMe7IQLRhAkT9NZbb2n79u2qX7++YzwqKkqSqs305OfnO2aNoqKiVFZWpoKCgkvWXIy/v79CQ0OdHgAA4Obk0YHIMAw98cQT2rRpk/7xj3+ocePGTssbN26sqKgopaenO8bKysq0Y8cOderUSZIUGxsrX19fp5rc3Fx98cUXjhoAAGBuHn2V2eOPP66NGzfqb3/7m0JCQhwzQVarVYGBgbJYLEpMTFRycrKaNWumZs2aKTk5WUFBQUpISHDUjhkzRpMnT1Z4eLjCwsI0ZcoUtWrVynHVGQAAMDePDkQrVqyQJMXFxTmNr127VqNGjZIkTZs2TSUlJRo/frwKCgrUoUMHbdmyRSEhIY76JUuWyMfHR0OGDFFJSYm6d++udevWydvb+3rtCgAA8GAeHYgMw/jZGovForlz52ru3LmXrAkICNCyZcu0bNkyF3YHAABuFh59DhEAAMD1QCACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACmRyACAACm5+PuBgAA+LHYqX9xdwseIXPhCHe3YCrMEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANPzcXcDAACgumPzWrm7BY/QYM7+6/I6zBABAADTIxABAADTIxABAADTIxABAADTIxABAADTIxABAADTIxABAADTM1Ugev7559W4cWMFBAQoNjZW//znP93dEgAA8ACmCUSvvvqqEhMTNWvWLH366afq0qWL+vbtq2PHjrm7NQAA4GamCUSLFy/WmDFj9Mgjj6hFixZaunSp7Ha7VqxY4e7WAACAm5ni1h1lZWXKzMzUjBkznMZ79eqlXbt2XXSd0tJSlZaWOp4XFhZKkoqKimrcR0VpSY3XvZmc9q1wdwse45d8nlyFz+UFfC4v4DPpOfhMXvBLP5NV6xuGcdk6UwSi//znP6qoqFBkZKTTeGRkpPLy8i66TkpKip5++ulq43a7/Zr0aCYx7m7Ak6RY3d0B/g+fy//DZ9Jj8Jn8Py76TJ4+fVpW66W3ZYpAVMVisTg9Nwyj2liVmTNnKikpyfG8srJSP/zwg8LDwy+5Dn5eUVGR7Ha7jh8/rtDQUHe3A0jicwnPw2fSdQzD0OnTp2Wz2S5bZ4pAVLduXXl7e1ebDcrPz682a1TF399f/v7+TmO1a9e+Vi2aTmhoKP/I4XH4XMLT8Jl0jcvNDFUxxUnVfn5+io2NVXp6utN4enq6OnXq5KauAACApzDFDJEkJSUlafjw4Wrfvr06duyoVatW6dixY/r973/v7tYAAICbmSYQDR06VN9//73mzZun3NxcxcTE6N1331XDhg3d3Zqp+Pv766mnnqp2OBJwJz6X8DR8Jq8/i/Fz16EBAADc5ExxDhEAAMDlEIgAAIDpEYgAAIDpEYhwTTRq1EhLly79RdsYNWqUBg4c6JJ+AAC4HAKRCY0aNUoWi8XxCA8PV58+ffT555+7uzWgmqrP68W+ImP8+PGyWCwaNWqUS17LYrHozTffdMm2gB/78e9dX19fNWnSRFOmTNGZM2d05MgRWSwWZWVlVVsvLi5OiYmJTs+rtuPv769f/epX6tevnzZt2nT9duYmRSAyqT59+ig3N1e5ubnatm2bfHx8FB8f7+62XMowDJ0/f97dbcAF7Ha7UlNTVVLy/2/6ee7cOb3yyitq0KCBGzu7uPLycne3AA9U9Xv3X//6l5555hk9//zzmjJlylVvZ+zYscrNzdWhQ4f0xhtv6Pbbb9dDDz2kRx999Bp0bR4EIpPy9/dXVFSUoqKi1LZtW02fPl3Hjx/XyZMnJUnTp09X8+bNFRQUpCZNmmj27NnVfsm/9dZbat++vQICAlS3bl0NHjzYafnZs2c1evRohYSEqEGDBlq1apXT8m+//VZDhw5VnTp1FB4ergEDBujIkSOX7Lm0tFQTJ05URESEAgIC1LlzZ2VkZDiWf/DBB7JYLHr//ffVvn17+fv765///Kc+++wzdevWTSEhIQoNDVVsbKz27t37C99BXE/t2rVTgwYNnP4XvGnTJtntdt1xxx2Osc2bN6tz586qXbu2wsPDFR8fr8OHDzuWl5WV6YknnlB0dLQCAgLUqFEjpaSkSLpwmFeSBg0aJIvF4nguSX//+98VGxurgIAANWnSRE8//bRT2LZYLHrhhRc0YMAABQcH65lnnlFBQYGGDRumevXqKTAwUM2aNdPatWuv0TuEG0HV71273a6EhAQNGzasRjOSQUFBju3cfffdmj9/vlauXKnVq1dr69atrm/cJAhEUHFxsV5++WU1bdpU4eHhkqSQkBCtW7dOBw4c0J/+9CetXr1aS5YscazzzjvvaPDgwbr//vv16aefatu2bWrfvr3TdhctWqT27dvr008/1fjx4/XYY4/pq6++knQhLHXr1k21atXShx9+qJ07d6pWrVrq06ePysrKLtrntGnT9MYbb2j9+vXat2+fmjZtqt69e+uHH36oVpeSkqLs7Gy1bt1aw4YNU/369ZWRkaHMzEzNmDFDvr6+rnwLcR08/PDDToHixRdf1OjRo51qzpw5o6SkJGVkZGjbtm3y8vLSoEGDVFlZKUn685//rLfeekuvvfaavv76a7300kuO4FMVrteuXavc3FzH8/fff1+/+93vNHHiRB04cEArV67UunXr9Oyzzzq99lNPPaUBAwZo//79Gj16tGbPnq0DBw7ovffeU3Z2tlasWKG6deteq7cHN6DAwECXzSaOHDlSderU4dDZL2HAdEaOHGl4e3sbwcHBRnBwsCHJiI6ONjIzMy+5zoIFC4zY2FjH844dOxrDhg27ZH3Dhg2N3/3ud47nlZWVRkREhLFixQrDMAxjzZo1xq233mpUVlY6akpLS43AwEDj/fffd/Q5YMAAwzAMo7i42PD19TVefvllR31ZWZlhs9mMBQsWGIZhGNu3bzckGW+++aZTLyEhIca6det+7m2Bh6r6HJw8edLw9/c3cnJyjCNHjhgBAQHGyZMnjQEDBhgjR4686Lr5+fmGJGP//v2GYRjGhAkTjPvuu8/pc/djkoy0tDSnsS5duhjJyclOYxs2bDCio6Od1ktMTHSq6devn/Hwww9f5d7iZvXj32eGYRiffPKJER4ebgwZMsTIyckxJBmBgYGO38tVDy8vL+PJJ590rNe1a1en5z/WoUMHo2/fvtd2R25iprl1B5x169ZNK1askCT98MMPev7559W3b1/t2bNHDRs21Ouvv66lS5fq0KFDKi4u1vnz553uuJyVlaWxY8de9jVat27t+NlisSgqKkr5+fmSpMzMTB06dEghISFO65w7d87pEEeVw4cPq7y8XPfcc49jzNfXV3fddZeys7Odan86U5WUlKRHHnlEGzZsUI8ePfTggw/qlltuuWzv8Dx169bV/fffr/Xr18swDN1///3VZlwOHz6s2bNna/fu3frPf/7jmBk6duyYYmJiNGrUKPXs2VO33nqr+vTpo/j4ePXq1euyr5uZmamMjAynGaGKigqdO3dOZ8+eVVBQkKTqn7vHHntMDzzwgPbt26devXpp4MCB3Eza5N5++23VqlVL58+fV3l5uQYMGKBly5bp7NmzkqRXX31VLVq0cFpn2LBhV7x9wzBksVhc2rOZEIhMKjg4WE2bNnU8j42NldVq1erVqxUfH6+HHnpITz/9tHr37i2r1arU1FQtWrTIUR8YGPizr/HTw1IWi8XxB6qyslKxsbF6+eWXq61Xr169amPG/91h5qf/2C/2CyA4ONjp+dy5c5WQkKB33nlH7733np566imlpqZq0KBBP7sP8CyjR4/WE088IUn6n//5n2rL+/XrJ7vdrtWrV8tms6myslIxMTGOw7Dt2rVTTk6O3nvvPW3dulVDhgxRjx499Prrr1/yNSsrK/X0009XO0dOkgICAhw///Rz17dvXx09elTvvPOOtm7dqu7du+vxxx/Xf//3f9do33Hjq/qPqK+vr2w2m+N3ZNW5k3a73en3snRlv2ulCyH9m2++0Z133unSns2EQARJF4KGl5eXSkpK9NFHH6lhw4aaNWuWY/nRo0ed6lu3bq1t27bp4YcfrtHrtWvXTq+++qoiIiKcZp4upWnTpvLz89POnTuVkJAg6cKVPHv37nW6JPVSmjdvrubNm2vSpEn67W9/q7Vr1xKIbkA/Psesd+/eTsu+//57ZWdna+XKlerSpYskaefOndW2ERoaqqFDh2ro0KH6zW9+oz59+uiHH35QWFiYfH19VVFR4VTfrl07ff3119X+UF2JevXqadSoURo1apS6dOmiqVOnEohM7Kf/EXWl9evXq6CgQA888MA12b4ZEIhMqrS0VHl5eZKkgoICLV++XMXFxerXr58KCwt17Ngxpaam6s4779Q777yjtLQ0p/Wfeuopde/eXbfccoseeughnT9/Xu+9956mTZt2Ra8/bNgwLVy4UAMGDNC8efNUv359HTt2TJs2bdLUqVNVv359p/rg4GA99thjmjp1qsLCwtSgQQMtWLBAZ8+e1ZgxYy75OiUlJZo6dap+85vfqHHjxjpx4oQyMjL4pXGD8vb2dhwi9fb2dlpWdbXiqlWrFB0drWPHjmnGjBlONUuWLFF0dLTatm0rLy8v/fWvf1VUVJRq164t6cKVZtu2bdM999wjf39/1alTR3PmzFF8fLzsdrsefPBBeXl56fPPP9f+/fv1zDPPXLLXOXPmKDY2Vi1btlRpaanefvvtaodDgJo4e/as8vLydP78eX377bfatGmTlixZoscee0zdunVzd3s3LK4yM6nNmzcrOjpa0dHR6tChgzIyMvTXv/5VcXFxGjBggCZNmqQnnnhCbdu21a5duzR79myn9ePi4vTXv/5Vb731ltq2bav77rtPn3zyyRW/flBQkD788EM1aNBAgwcPVosWLTR69GiVlJRccsboueee0wMPPKDhw4erXbt2OnTokN5//33VqVPnkq/j7e2t77//XiNGjFDz5s01ZMgQ9e3bV08//fQV9wrPEhoaetHPiJeXl1JTU5WZmamYmBhNmjRJCxcudKqpVauW5s+fr/bt2+vOO+/UkSNH9O6778rL68KvwkWLFik9Pd3pcv7evXvr7bffVnp6uu68807dfffdWrx4sRo2bHjZPv38/DRz5ky1bt1a9957r7y9vZWamuqidwFmtnr1akVHR+uWW27RoEGDdODAAb366qt6/vnn3d3aDc1iVJ2cAQAAYFLMEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAEAANMjEAG4IVksFr355pvubkOjRo3SwIED3d0GgF+IQATA7UaNGiWLxVLt0adPH3e35nDkyBFZLBZlZWU5jf/pT3/SunXr3NITANfh5q4APEKfPn20du1apzF/f383dXPlrFaru1sA4ALMEAHwCP7+/oqKinJ6VN2495tvvtG9996rgIAA3X777UpPT3da94MPPpDFYtGpU6ccY1lZWbJYLDpy5Ihj7KOPPlLXrl0VFBSkOnXqqHfv3iooKJB04YbHnTt3Vu3atRUeHq74+HgdPnzYsW7jxo0lSXfccYcsFovi4uIkVT9kVlpaqokTJyoiIkIBAQHq3LmzMjIyqvW6bds2tW/fXkFBQerUqZO+/vprV7yNAGqIQATAo1VWVmrw4MHy9vbW7t279cILL2j69OlXvZ2srCx1795dLVu21Mcff6ydO3eqX79+qqiokCSdOXNGSUlJysjI0LZt2+Tl5aVBgwapsrJSkrRnzx5J0tatW5Wbm6tNmzZd9HWmTZumN954Q+vXr9e+ffvUtGlT9e7dWz/88INT3axZs7Ro0SLt3btXPj4+Gj169FXvEwDX4ZAZAI/w9ttvq1atWk5j06dPV4cOHZSdna0jR46ofv36kqTk5GT17dv3qra/YMECtW/fXs8//7xjrGXLlo6fH3jgAaf6NWvWKCIiQgcOHFBMTIzq1asnSQoPD1dUVNRFX+PMmTNasWKF1q1b5+hv9erVSk9P15o1azR16lRH7bPPPquuXbtKkmbMmKH7779f586dU0BAwFXtFwDXYIYIgEfo1q2bsrKynB6PP/64srOz1aBBA0cYkqSOHTte9farZogu5fDhw0pISFCTJk0UGhrqOER27NixK36Nw4cPq7y8XPfcc49jzNfXV3fddZeys7Odalu3bu34OTo6WpKUn59/xa8FwLWYIQLgEYKDg9W0adNq44ZhVBuzWCxOz728vKrVlpeXO9UEBgZe9vX79esnu92u1atXy2azqbKyUjExMSorK7vifah6/Z/2ZxhGtTFfX1/Hz1XLqg7PAbj+mCEC4NFuv/12HTt2TN99951j7OOPP3aqqTqclZub6xj76eXxrVu31rZt2y76Gt9//72ys7P1X//1X+revbtatGjhONm6ip+fnyQ5zjm6mKZNm8rPz087d+50jJWXl2vv3r1q0aLFZfYSgLsxQwTAI5SWliovL89pzMfHRz169NCtt96qESNGaNGiRSoqKtKsWbOc6po2bSq73a65c+fqmWee0TfffKNFixY51cycOVOtWrXS+PHj9fvf/15+fn7avn27HnzwQYWFhSk8PFyrVq1SdHS0jh07phkzZjitHxERocDAQG3evFn169dXQEBAtUvug4OD9dhjj2nq1KkKCwtTgwYNtGDBAp09e1Zjxoxx4bsFwNWYIQLgETZv3qzo6GinR+fOneXl5aW0tDSVlpbqrrvu0iOPPKJnn33WaV1fX1+98sor+uqrr9SmTRvNnz9fzzzzjFNN8+bNtWXLFn322We666671LFjR/3tb3+Tj4+PvLy8lJqaqszMTMXExGjSpElauHCh0/o+Pj7685//rJUrV8pms2nAgAEX3Y/nnntODzzwgIYPH6527drp0KFDev/99x1fIQDAM1mMix2gBwAAMBFmiAAAgOkRiAAAgOkRiAAAgOkRiAAAgOkRiAAAgOkRiAAAgOkRiAAAgOkRiAAAgOkRiAAAgOkRiAAAgOkRiAAAgOn9P5bZY3TQF+afAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data = data1 ,x='Education',hue='LeaveOrNot')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fa9ff728",
   "metadata": {},
   "source": [
    "## most of the employees arebachelors leaving the company"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "adc3148f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\91772\\AppData\\Local\\Temp\\ipykernel_10040\\1427453235.py:2: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  data1['AgeGroup'] = pd.qcut(data1['Age'], q=3, labels=groups)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='AgeGroup', ylabel='count'>"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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Zx3ncDic1o0x5dpOd74xkydPVn5+vrUcOHDgwooHAABXDJeFnfT0dOXm5qpDhw5yd3eXu7u7UlNT9Y9//EPu7u7WjM7pMzS5ublWX0hIiIqLi5WXl3fWMWfi5eWlgIAApwUAANiTy8JOz549tX37dmVkZFhLx44dNWzYMGVkZOjaa69VSEiI1q9fb21TXFys1NRUde3aVZLUoUMHeXh4OI3Jzs7Wjh07rDEAAODq5rJ7dvz9/RUZGenU5uvrq6CgIKs9Li5OCQkJioiIUEREhBISElSzZk0NHTpUkhQYGKiYmBiNHz9eQUFBqlOnjiZMmKBWrVqVu+EZAABcnVx6g/L5TJw4UYWFhYqNjVVeXp46d+6sdevWyd/f3xozd+5cubu7a8iQISosLFTPnj21dOlSubm5ubByAABQXVSrsLNx40andYfDofj4eMXHx591G29vb82bN0/z5s27tMUBAIArksufswMAAHApEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtVSrs3HLLLTp8+HC59oKCAt1yyy0XWxMAAECVqVTY2bhxo4qLi8u1//rrr/r4448vuigAAICq4n4hg7/66ivrz7t27VJOTo61XlpaqrVr1+qaa66puuoAAAAu0gWFnbZt28rhcMjhcJzxcpWPj4/mzZtXZcUBAABcrAsKO1lZWTLG6Nprr9XWrVtVr149q8/T01P169eXm5tblRcJAABQWRcUdho3bixJKisruyTFAAAAVLULCju/tWfPHm3cuFG5ubnlws/UqVMvujAAAICqUKmw8/LLL+vBBx9U3bp1FRISIofDYfU5HA7CDgAAqDYqFXaefvppzZgxQ5MmTarqegAAAKpUpZ6zk5eXpzvvvLOqawEAAKhylQo7d955p9atW1fVtQAAAFS5Sl3Gatq0qZ588klt2bJFrVq1koeHh1P/2LFjq6Q4AACAi1WpsLNo0SL5+fkpNTVVqampTn0Oh4OwAwAAqo1KhZ2srKyqrgMAAOCSqNQ9OwAAAFeKSs3sjBo16pz9ixcvrlQxAAAAVa1SYScvL89pvaSkRDt27NDhw4fP+AWhAAAArlKpsJOSklKuraysTLGxsbr22msvuigAAICq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Dw/XDTfcIC8vLzVo0ECPP/64Tpw4YW17pqnatm3bKj4+3lp3OBz65z//qdtvv101a9ZURESE3nrrLadt3nrrLUVERMjHx0c333yzli1bJofDocOHD1/CI0dVat++vRo1aqQ1a9ZYbWvWrFFYWJjatWtntZ0+fZ6bm6sBAwbIx8dH4eHheu2118rte+/everevbu8vb113XXXlXti+pns2rVL/fr1k5+fn4KDgzV8+HD9/PPPTmOOHTumVatW6cEHH1T//v21dOnScvupyLmZlpam7t27y8fHR2FhYRo7dqyOHTt2QceIy+d8v/dnnnlGwcHB8vf3V0xMjH799Ven/jNdAho0aJA1e3m6Jk2aSJJuv/12ORwOa/2UV199VU2aNFFgYKDuvvtuHTlyxOorKirS2LFjVb9+fXl7e+vGG2/Utm3bnLZPTU095/v0sWPHdM8998jPz08NGjTQ7Nmzz/9DshnCDiRJI0aMUM+ePfWXv/xF8+fP144dO/T888+rX79+6tSpk7788kstXLhQSUlJevrppy94/0899ZSGDBmir776Sv369dOwYcN06NAhSSfvt/jjH/+oQYMGKSMjQ6NHj9aUKVOq+hBxGfzlL3/RkiVLrPXFixdr1KhR59xm5MiR2rdvnz766CO9/vrrWrBggXJzc63+srIyDR48WG5ubtqyZYtefPFFTZo06Zz7zM7OVo8ePdS2bVt99tlnWrt2rX788UcNGTLEadzKlSvVvHlzNW/eXH/+85+1ZMkS/fbRYxU5N7dv367o6GgNHjxYX331lVauXKlNmzbp4YcfrvAx4vI61+991apVmjZtmmbMmKHPPvtMDRo00IIFCy7q9U6FkyVLlig7O9sprHz77bd644039Pbbb+vtt99WamqqnnnmGat/4sSJWr16tZYtW6bPP/9cTZs2VXR0tPX++d///ve879OPPfaYNmzYoJSUFK1bt04bN25Uenr6RR3TFccA/+fHH3809erVMzVq1DBr1qwxTzzxhGnevLkpKyuzxrzwwgvGz8/PlJaWGmOMady4sZk7d67Tftq0aWOmTZtmrUsyf/3rX631o0ePGofDYd577z1jjDGTJk0ykZGRTvuYMmWKkWTy8vKq9iBxSYwYMcLcdttt5qeffjJeXl4mKyvL7Nu3z3h7e5uffvrJ3HbbbWbEiBHGGGN69OhhHnnkEWOMMbt37zaSzJYtW6x9ZWZmGknWefX+++8bNzc3c+DAAWvMe++9ZySZlJQUY4wxWVlZRpL54osvjDHGPPnkk6ZPnz5ONR44cMBIMrt377baunbtahITE40xxpSUlJi6deua9evXW/0VOTeHDx9u7r//fqcxH3/8salRo4YpLCys0DHi8jrX771Lly7mgQcecBrfuXNn06ZNG2v9t+fwKb89x40p/9742/P1lGnTppmaNWuagoICq+2xxx4znTt3NsacfK/08PAwr732mtVfXFxsQkNDzXPPPWeMMed9nz5y5Ijx9PQ0ycnJVv8vv/xifHx8yh2DnTGzA0v9+vV1//33q2XLlrr99tuVmZmpLl26yOFwWGO6deumo0eP6uDBgxe079atW1t/9vX1lb+/v/U/2927d6tTp05O42+44YaLOBK4St26dfWHP/xBy5Yt05IlS/SHP/xBdevWPev4zMxMubu7q2PHjlZbixYtVKtWLacxjRo1UsOGDa22Ll26nLOO9PR0bdiwQX5+ftbSokULSSf/Jy2dPO+2bt2qu+++W5Lk7u6uu+66S4sXL7b2U5FzMz09XUuXLnV6rejoaJWVlSkrK6tCx4jL53y/91Pve791vvPtYjRp0sTpHrYGDRpY743ffvutSkpK1K1bN6vfw8NDN9xwgzIzM53qPdv79Lfffqvi4mKnY6hTp46aN29+yY6pOrLFd2Oh6ri7u8vd/eRpYYxx+gt0qk2S1V6jRg2naX9JKikpKbdfDw8Pp3WHw6GysrLzvg6uPKNGjbIu4bzwwgvnHHv6+XSuMb91rvHSyUtfAwYM0LPPPluur0GDBpKkpKQknThxQtdcc43Ta3l4eCgvL0+1a9eu0LlZVlam0aNHa+zYseVeq1GjRtq9e3eFasblcb7fe0VU9H2vIs733niq7bd+e16e732a99KTmNnBWV133XVKS0tz+suSlpYmf39/642iXr16ys7OtvoLCgqUlZV1Qa/TokWLcjfcffbZZxdROVzp1ltvVXFxsYqLi8/4SIPfatmypU6cOOH0+969e7fTzb/XXXed9u/frx9++MFq++STT8653/bt22vnzp1q0qSJmjZt6rT4+vrqxIkTeuWVVzR79mxlZGRYy5dffqnGjRtbNxBX5Nw89Vqnv07Tpk3l6elZoWPE5VGR33vLli21ZcsWp+1OXz/9fa+0tFQ7duw452t7eHiotLT0guo9dQ5t2rTJaispKdFnn32mli1bSjr/+3TTpk3l4eHhdAx5eXnas2fPBdVypSPs4KxiY2N14MABjRkzRl9//bXefPNNTZs2TePGjVONGidPnVtuuUWvvvqqPv74Y+3YsUMjRoyQm5vbBb3O6NGj9fXXX2vSpEnas2ePVq1aZX06gv8NX3nc3NyUmZmpzMzM854LzZs316233qr77rtPn376qdLT03XvvffKx8fHGtOrVy81b95c99xzj7788kt9/PHH572B/aGHHtKhQ4f0pz/9SVu3btV3332ndevWadSoUSotLdXbb7+tvLw8xcTEKDIy0mn54x//qKSkJEkVOzcnTZqkTz75RA899JAyMjK0d+9evfXWWxozZkyFjxGXR0V+74888ogWL16sxYsXa8+ePZo2bZp27tzptJ9bbrlF77zzjt555x19/fXXio2NPW94bdKkiT788EPl5ORUeAbJ19dXDz74oB577DGtXbtWu3bt0n333afjx48rJiZG0vnfp/38/BQTE6PHHntMH374oXbs2KGRI0da7+FXi6vraHFBrrnmGr377rvaunWr2rRpowceeEAxMTH661//ao2ZPHmyunfvrv79+6tfv34aNGiQfve7313Q64SHh+v111/XmjVr1Lp1ay1cuND6x8zLy6tKjwmXR0BAgAICAio0dsmSJQoLC1OPHj00ePBg3X///apfv77VX6NGDaWkpKioqEg33HCD7r33Xs2YMeOc+wwNDdXmzZtVWlqq6OhoRUZG6pFHHlFgYKBq1KihpKQk9erVS4GBgeW2veOOO5SRkaHPP/+8Qudm69atlZqaqr179+qmm25Su3bt9OSTT1qXyypyjLg8KvJ7j4iI0NSpUzVp0iR16NBB33//vR588EGnsaNGjdKIESN0zz33qEePHgoPD9fNN998zteePXu21q9fX+5RDOfzzDPP6I477tDw4cPVvn17ffPNN3r//fdVu3ZtSRV7n541a5a6d++ugQMHqlevXrrxxhvVoUOHCtdgBw7DBT1UQzNmzNCLL76oAwcOuLoUwAnnJnDl4QZlVAsLFixQp06dFBQUpM2bN2vWrFlOzykBXIVzE7jyEXZQLezdu1dPP/20Dh06pEaNGmn8+PGaPHmyq8sCODcBG+AyFgAAsDVuUAYAALZG2AEAALZG2AEAALZG2AEAALZG2AEAALZG2AEAALZG2AFwyaWlpcnNzU233nrrJXuNb775RqNGjVKjRo3k5eWla665Rj179tRrr72mEydOXLLXBVD9EXYAXHKLFy/WmDFjtGnTJu3fv7/K979161a1b99emZmZeuGFF7Rjxw69/fbbGjVqlF588cVyX+T4WyUlJVVeD4DqhbAD4JI6duyYVq1apQcffFD9+/e3vjX8lLfeeksRERHy8fHRzTffrGXLlsnhcDh9i3RaWpq6d+8uHx8fhYWFaezYsTp27JgkyRijkSNHqlmzZtq8ebMGDBigiIgItWvXTsOGDdPHH3+s1q1bS5L27dsnh8OhVatWKSoqSt7e3lq+fLnKyso0ffp0NWzYUF5eXmrbtq3Wrl1rvf7GjRvL1ZSRkSGHw6F9+/ZJkpYuXapatWrpjTfeULNmzeTt7a3evXvzHVpANUDYAXBJrVy5Us2bN1fz5s315z//WUuWLNGpB7fv27dPf/zjHzVo0CBlZGRo9OjR1reKn7J9+3ZFR0dr8ODB+uqrr7Ry5Upt2rTJ+n6qjIwMZWZmasKECapR48xvaQ6Hw2l90qRJGjt2rDIzMxUdHa3nn39es2fP1t///nd99dVXio6O1sCBA7V3794LOtbjx49rxowZWrZsmTZv3qyCggLdfffdF7QPAJeAAYBLqGvXriYxMdEYY0xJSYmpW7euWb9+vTHGmEmTJpnIyEin8VOmTDGSTF5enjHGmOHDh5v777/faczHH39satSoYQoLC01ycrKRZD7//HOr/8cffzS+vr7W8sILLxhjjMnKyjKSrHpOCQ0NNTNmzHBq69Spk4mNjTXGGLNhwwanmowx5osvvjCSTFZWljHGmCVLlhhJZsuWLdaYzMxMI8l8+umnF/IjA1DFmNkBcMns3r1bW7dutWY33N3dddddd2nx4sVWf6dOnZy2ueGGG5zW09PTtXTpUvn5+VlLdHS0ysrKlJWVZY377exNUFCQMjIylJGRoVq1aqm4uNhpnx07drT+XFBQoB9++EHdunVzGtOtWzdlZmZe0PG6u7s77btFixaqVavWBe8HQNXiW88BXDJJSUk6ceKErrnmGqvNGCMPDw/l5eXJGFPuEpM57buJy8rKNHr0aI0dO7bc/hs1aqTCwkJJ0tdff622bdtKktzc3NS0aVNJJwPI6Xx9fcu1namOU22nLo/9traz3dh8+n7O1gbg8mFmB8AlceLECb3yyiuaPXu2NcuSkZGhL7/8Uo0bN9Zrr72mFi1aaNu2bU7bffbZZ07r7du3186dO9W0adNyi6enp9q1a6cWLVro73//u8rKyi64zoCAAIWGhmrTpk1O7WlpaWrZsqUkqV69epKk7Oxsqz8jI+OMx/zb+nfv3q3Dhw+rRYsWF1wXgCrk0otoAGwrJSXFeHp6msOHD5fre+KJJ0zbtm3Nd999Zzw8PMzEiRPN7t27zcqVK03Dhg2NJGu7L7/80vj4+JjY2FjzxRdfmD179pg333zTPPzww9b+PvnkE+Pn52d+//vfmzfffNPs2bPH7Ny50yxcuNDUrFnT/OMf/zDG/O+enS+++MKpnrlz55qAgACTnJxsvv76azNp0iTj4eFh9uzZY4wxpri42ISFhZk777zT7N6927z99tumefPm5e7Z8fDwMDfccIPZsmWLSU9PN126dDG///3vL8FPF8CFIOwAuCT69+9v+vXrd8a+9PR0I8mkp6ebN9980zRt2tR4eXmZqKgos3DhQiPJFBYWWuO3bt1qevfubfz8/Iyvr69p3bp1uRuKd+/ebUaMGGEaNmxo3N3dTWBgoOnevbt56aWXTElJiTHm7GGntLTUPPXUU+aaa64xHh4epk2bNua9995zGrNp0ybTqlUr4+3tbW666Sbz//7f/ysXdgIDA83q1avNtddeazw9Pc0tt9xi9u3bd5E/SQAXy2HMaRfIAcCFZsyYoRdffPGKez7N0qVLFRcX5/QsHgDVAzcoA3CpBQsWqFOnTgoKCtLmzZs1a9Ys6xk6AFAVCDsAXGrv3r16+umndejQITVq1Ejjx4/X5MmTXV0WABvhMhYAALA1PnoOAABsjbADAABsjbADAABsjbADAABsjbADAABsjbADAABsjbADAABsjbADAABs7f8DR9lHwh/1OioAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "groups = ['Young', 'MiddleAged', 'Adulthood']\n",
    "data1['AgeGroup'] = pd.qcut(data1['Age'], q=3, labels=groups)\n",
    "sns.countplot(data = data1 ,x='AgeGroup',hue='LeaveOrNot')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "98f6b9e8",
   "metadata": {},
   "source": [
    "# most of young employees are leaving the comapany`"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "00c64166",
   "metadata": {},
   "source": [
    "## Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "c1949056",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\91772\\anconda\\lib\\site-packages\\pandas\\core\\algorithms.py:798: FutureWarning: In a future version, the Index constructor will not infer numeric dtypes when passed object-dtype sequences (matching Series behavior)\n",
      "  uniques = Index(uniques)\n"
     ]
    }
   ],
   "source": [
    "data2 = pd.get_dummies(data1, columns = ['AgeGroup','EverBenched','City','JoiningYear','Education','Gender'])\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "9dc183a5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PaymentTier</th>\n",
       "      <th>Age</th>\n",
       "      <th>ExperienceInCurrentDomain</th>\n",
       "      <th>LeaveOrNot</th>\n",
       "      <th>AgeGroup_Young</th>\n",
       "      <th>AgeGroup_MiddleAged</th>\n",
       "      <th>AgeGroup_Adulthood</th>\n",
       "      <th>EverBenched_No</th>\n",
       "      <th>EverBenched_Yes</th>\n",
       "      <th>City_Bangalore</th>\n",
       "      <th>...</th>\n",
       "      <th>JoiningYear_2014</th>\n",
       "      <th>JoiningYear_2015</th>\n",
       "      <th>JoiningYear_2016</th>\n",
       "      <th>JoiningYear_2017</th>\n",
       "      <th>JoiningYear_2018</th>\n",
       "      <th>Education_Bachelors</th>\n",
       "      <th>Education_Masters</th>\n",
       "      <th>Education_PHD</th>\n",
       "      <th>Gender_Female</th>\n",
       "      <th>Gender_Male</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>34</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>28</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>38</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>27</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>24</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 24 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "  PaymentTier  Age  ExperienceInCurrentDomain  LeaveOrNot  AgeGroup_Young  \\\n",
       "0           3   34                          0           0               0   \n",
       "1           1   28                          3           1               1   \n",
       "2           3   38                          2           0               0   \n",
       "3           3   27                          5           1               1   \n",
       "4           3   24                          2           1               1   \n",
       "\n",
       "   AgeGroup_MiddleAged  AgeGroup_Adulthood  EverBenched_No  EverBenched_Yes  \\\n",
       "0                    0                   1               1                0   \n",
       "1                    0                   0               1                0   \n",
       "2                    0                   1               1                0   \n",
       "3                    0                   0               1                0   \n",
       "4                    0                   0               0                1   \n",
       "\n",
       "   City_Bangalore  ...  JoiningYear_2014  JoiningYear_2015  JoiningYear_2016  \\\n",
       "0               1  ...                 0                 0                 0   \n",
       "1               0  ...                 0                 0                 0   \n",
       "2               0  ...                 1                 0                 0   \n",
       "3               1  ...                 0                 0                 1   \n",
       "4               0  ...                 0                 0                 0   \n",
       "\n",
       "   JoiningYear_2017  JoiningYear_2018  Education_Bachelors  Education_Masters  \\\n",
       "0                 1                 0                    1                  0   \n",
       "1                 0                 0                    1                  0   \n",
       "2                 0                 0                    1                  0   \n",
       "3                 0                 0                    0                  1   \n",
       "4                 1                 0                    0                  1   \n",
       "\n",
       "   Education_PHD  Gender_Female  Gender_Male  \n",
       "0              0              0            1  \n",
       "1              0              1            0  \n",
       "2              0              1            0  \n",
       "3              0              0            1  \n",
       "4              0              0            1  \n",
       "\n",
       "[5 rows x 24 columns]"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "ec2e126b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2211, 23) (553, 23) (2211,) (553,)\n"
     ]
    }
   ],
   "source": [
    "X = data2.drop('LeaveOrNot',axis=1)\n",
    "y= data2.LeaveOrNot.values\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)\n",
    "\n",
    "print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "13ad1fa2",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test =np.ascontiguousarray(X_test)\n",
    "X_train=np.ascontiguousarray(X_train)\n",
    "y_train=np.ascontiguousarray(y_train)\n",
    "y_test=np.ascontiguousarray(y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "49e2cd8e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-3\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>KNeighborsClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-11\" type=\"checkbox\" checked><label for=\"sk-estimator-id-11\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">KNeighborsClassifier</label><div class=\"sk-toggleable__content\"><pre>KNeighborsClassifier()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "KNeighborsClassifier()"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "knc = KNeighborsClassifier()\n",
    "\n",
    "knc.fit(X_train,y_train)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "431e33c2",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\91772\\anconda\\lib\\site-packages\\xgboost\\sklearn.py:1395: UserWarning: `use_label_encoder` is deprecated in 1.7.0.\n",
      "  warnings.warn(\"`use_label_encoder` is deprecated in 1.7.0.\")\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Algorithm</th>\n",
       "      <th>Accuracy</th>\n",
       "      <th>Precision</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>lgbm</td>\n",
       "      <td>0.831826</td>\n",
       "      <td>0.926667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>DT</td>\n",
       "      <td>0.828210</td>\n",
       "      <td>0.893750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>AdaBoost</td>\n",
       "      <td>0.755877</td>\n",
       "      <td>0.890909</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>xgb</td>\n",
       "      <td>0.826401</td>\n",
       "      <td>0.878788</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>GBDT</td>\n",
       "      <td>0.755877</td>\n",
       "      <td>0.864407</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>RF</td>\n",
       "      <td>0.801085</td>\n",
       "      <td>0.858065</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>LR</td>\n",
       "      <td>0.770344</td>\n",
       "      <td>0.797468</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>NB</td>\n",
       "      <td>0.755877</td>\n",
       "      <td>0.786667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>BgC</td>\n",
       "      <td>0.726944</td>\n",
       "      <td>0.666667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>ETC</td>\n",
       "      <td>0.712477</td>\n",
       "      <td>0.642202</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>KN</td>\n",
       "      <td>0.698011</td>\n",
       "      <td>0.635000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Algorithm  Accuracy  Precision\n",
       "10      lgbm  0.831826   0.926667\n",
       "2         DT  0.828210   0.893750\n",
       "5   AdaBoost  0.755877   0.890909\n",
       "9        xgb  0.826401   0.878788\n",
       "8       GBDT  0.755877   0.864407\n",
       "4         RF  0.801085   0.858065\n",
       "3         LR  0.770344   0.797468\n",
       "1         NB  0.755877   0.786667\n",
       "6        BgC  0.726944   0.666667\n",
       "7        ETC  0.712477   0.642202\n",
       "0         KN  0.698011   0.635000"
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "knc = KNeighborsClassifier()\n",
    "mnb = MultinomialNB()\n",
    "dtc = DecisionTreeClassifier(max_depth=7,random_state=2)\n",
    "lrc = LogisticRegression(solver='liblinear', penalty='l1')\n",
    "rfc = RandomForestClassifier(n_estimators=17, random_state=2,max_depth=5)\n",
    "abc = AdaBoostClassifier(n_estimators=17, random_state=2,learning_rate=0.2)\n",
    "bc = BaggingClassifier(n_estimators=17, random_state=2)\n",
    "etc = ExtraTreesClassifier(n_estimators=50, random_state=2)\n",
    "gbdt = GradientBoostingClassifier(n_estimators=18,random_state=2)\n",
    "xgb = XGBClassifier(n_estimators=17,random_state=2,use_label_encoder=False,eval_metric='mlogloss')\n",
    "lgbm= LGBMClassifier(verbose=-1,\n",
    "                          learning_rate=0.1,\n",
    "                          max_depth=6,\n",
    "                          num_leaves=10, \n",
    "                          n_estimators=17,\n",
    "                          max_bin=500,random_state=2)\n",
    "\n",
    "\n",
    "clfs = {\n",
    "    'KN' : knc, \n",
    "    'NB': mnb, \n",
    "    'DT': dtc, \n",
    "    'LR': lrc, \n",
    "    'RF': rfc, \n",
    "    'AdaBoost': abc, \n",
    "    'BgC': bc, \n",
    "    'ETC': etc,\n",
    "    'GBDT':gbdt,\n",
    "    'xgb':xgb,\n",
    "    'lgbm':lgbm\n",
    "}\n",
    "\n",
    "def train_classifier(clf,X_train,y_train,X_test,y_test):\n",
    "    clf.fit(X_train,y_train)\n",
    "    y_pred = clf.predict(X_test)\n",
    "    accuracy = accuracy_score(y_test,y_pred)\n",
    "    precision = precision_score(y_test,y_pred,zero_division=0)\n",
    "    \n",
    "    return accuracy,precision\n",
    "\n",
    "accuracy_scores = []\n",
    "precision_scores = []\n",
    "\n",
    "for name,clf in clfs.items():\n",
    "    \n",
    "    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)\n",
    "    \n",
    "#     print(\"For \",name)\n",
    "#     print(\"Accuracy - \",current_accuracy)\n",
    "#     print(\"Precision - \",current_precision)\n",
    "    \n",
    "    accuracy_scores.append(current_accuracy)\n",
    "    precision_scores.append(current_precision)\n",
    "    \n",
    "\n",
    "performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)\n",
    "performance_df\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
