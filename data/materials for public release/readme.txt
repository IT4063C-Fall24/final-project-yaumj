PEW RESEARCH CENTER 
STEM Survey
July 11 - August 10, 2017
N=4,914


**************************************************************************************************************************

This survey was conducted by the GfK group in English and Spanish using KnowledgePanel, a nationally representative online research panel.
 
***************************************************************************************************************************

To protect the privacy of respondents, state has been removed from the public data file.

***************************************************************************************************************************

Releases from this survey:

January 9, 2018, Women and Men in STEM Often at Odds Over Workplace Equity
http://www.pewsocialtrends.org/2018/01/09/women-and-men-in-stem-often-at-odds-over-workplace-equity/


***************************************************************************************************************************
NOTES

WORKTYPE_FINAL change: Most respondents were first asked if their main job had changed since the time of completing a profile survey. If their job had changed, they were asked their occupation. However, some respondents saw the original reconfirmation question, which simply asked the occupation question again without asking if their job had changed. This most recent response was used to determine if they currently worked in a STEM or non-STEM job. For more, see the codebook.

Variable Confirmation_flag captures whether the respondent received the original reconfirmation questions or the new reconfirmation questions.

SCH10b change: Most respondents saw the following for SCH10B_5: I didn’t have enough support at home or after school to help me do well in these classes". But some respondents saw the original text: I didn’t have enough support at home or after school to do well in these classes.    

Variable SCH10_flag captures whether the respondent received the orginal SCH10B_5 or the new SCH10B_5. 


****************************************************************************************************************************

Created variable syntax - SPSS


**generate STEM_degree variable

compute STEM_DEGREE=$sysmis.
if (DEGREE1_OE1=997) or (DEGREE1_OE1=998) or (DEGREE1_OE1=999) STEM_DEGREE=9.
if (DEGREE1_OE2=997) or (DEGREE1_OE2=998) or (DEGREE1_OE2=999) STEM_DEGREE=9.
if (DEGREE2_OE=997) or (DEGREE2_OE=998) or (DEGREE2_OE=999) STEM_DEGREE=9.
if (DEGREE1_OE1>=211) and (DEGREE1_OE1<=300) STEM_DEGREE=2. 
if (DEGREE1_OE2>=211) and (DEGREE1_OE2<=300) STEM_DEGREE=2.
if (DEGREE2_OE>=211) and (DEGREE2_OE<=300) STEM_DEGREE=2.  
if (DEGREE1_OE1>=111) and (DEGREE1_OE1<=170) STEM_DEGREE=1.
if (DEGREE1_OE2>=111) and (DEGREE1_OE2<=170) STEM_DEGREE=1.
if (DEGREE2_OE>=111) and (DEGREE2_OE<=170) STEM_DEGREE=1.
execute.

variable labels STEM_DEGREE 'Respondent holds degree in STEM field or not'.
value labels STEM_DEGREE
1 'at least one STEM degree'
2 'no STEM degrees'
9 "Not enough information/Don't know/Refused".
execute.
