import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from typing import Optional

MAIN_FILE_LOC="images"
TITLE_FONT_SIZE=18
tl=pd.read_csv("../data/testData/AugmentedTrainingLogTestData.csv")
et=pd.read_csv("../data/testData/ExerciseTypeTestData.csv")
e=pd.read_csv("../data/testData/ExerciseTestData.csv")
units={
    "Volume": "lbs",
    "Effort": "RPE",
    "Intensity": "% of 1RM",
}

def listToFilter(_in: List[str]) -> str:
    rv=""
    for i,item in enumerate(_in):
        rv+=("'{0}'".format(item))
        if i+1<len(_in):
            rv+=","
    return rv

def getUnit(_in: str) -> str:
    return "({0})".format(units[_in]) if _in in units else ""

class Ch3(object):
    fileLoc="{0}/ch3".format(MAIN_FILE_LOC)
    __mergedDf=pd.DataFrame()

    @property
    def mergedDf(self):
        if Ch3.__mergedDf.empty:
            tmp=pd.merge(tl,e,left_on="ExerciseID", right_on="Id", how="outer")
            Ch3.__mergedDf=pd.merge(
                tmp,et,left_on="TypeID", right_on="Id", how="outer",
            )
        return Ch3.__mergedDf

    @staticmethod
    def oneVTwo(one: str,
            two: str,
            trendline: Optional[bool]=True,
            types: Optional[List[str]]=["Main Compound","Main Compound Accessory"]
        ) -> None:
        tmp=Ch3().mergedDf.query(
            "Intensity<=1.2 and (T in ({0}))".format(listToFilter(types)),
        )[[one,two]].dropna()

        if trendline:
            model=np.poly1d(np.polyfit(tmp[one],tmp[two],3))
            line=np.arange(0,max(tmp[one]),max(tmp[one])/1000)

        fig,ax=plt.subplots(figsize=(12,6))
        ax.scatter(tmp[one],tmp[two],s=20,alpha=0.5)
        if trendline:
            ax.plot(line,model(line))

        plt.ylim(0,max(tmp[two]))
        plt.title(
            "{0} vs {1} for Main Compound and Main Compound Accessories".format(
                one,two),
        fontsize=TITLE_FONT_SIZE)
        plt.xlabel("{0} {1}".format(one,getUnit(one)))
        plt.ylabel("{0} {1}".format(two,getUnit(two)))
        plt.grid()
        plt.savefig("{0}/{1}Vs{2}.png".format(Ch3.fileLoc,one,two),
            bbox_inches="tight"
        )


if __name__=="__main__":
    Ch3.oneVTwo("Intensity","Volume")
    Ch3.oneVTwo("Effort","Volume",False)
    Ch3.oneVTwo("Intensity","Effort")
    Ch3.oneVTwo("Sets","Reps",False)
