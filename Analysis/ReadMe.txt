Which folder contains which analysis?

-Comparing Answers:
	Selects all files which are an answer given by the presenter. Analysis regarding Academic Status, 
	Sex and Native Speaker. Plots, Logistic Regression, and Correlation (CHI2 and ANOVA not yet functional).

-Comparing_presentation_answer:
	Selects all files which er an answer but also files given by the presenter during the presentation. Analysis
	regards differences of emotion during answer session to during the presentation. Plots, Logistic Regression, 
	and Correlation (CHI2 and ANOVA not yet functional).

-Comparing Questions:
	Selects all files which are a question. Analysis regarding Academic Status, 
	Sex and Native Speaker. Plots, Logistic Regression, and Correlation (CHI2 and ANOVA not yet functional).

-Comparing_short_vs_long_samples:
	Compares UIST2019 Audio files - one data set containing OpenSMILE results from audio files with a length between 
	5 seconds and 2 minutes, another data set containing results from 5 to 30 second audio files. Compares the
	results from OpenSMILE regarding the effect of longer audio files on the result of the classification.

-General_Analysis_CHI_2019:
	Does not differ between questions, answers and sound files during presentation, but analyzes the whole dataset.
	Analysis regarding Academic Status, Sex and Native Speaker. Plots, Logistic Regression, CHI2, Ruskal-Wallis and Correlation.

-Notebooks_UIST2019/_short_samples:
	Will be removed in future, since they are combined in analysis in folder 'comparing_short_vs_long_samples'.

-OpenSMILE Data Analysis.ipynb:
	Prints meta information of the data set: Number of females/males in Questions, Answers, Presentation, General. 
	Same for other character features as Academic Status and NativeSpeaker
