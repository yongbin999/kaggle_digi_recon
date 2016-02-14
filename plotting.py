import numpy as np
import matplotlib.pyplot as plt

def bar_graph(title, name1,name2,name3, x_data,y_data, maxy=None,ylabel=None):
	#Create values and labels for bar chart
	values = y_data
	inds   = x_data
	labels = [name1,name2,name3]

	#Plot a bar chart
	plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot
	plt.bar(inds, values, align='center') #This plots the data
	plt.grid(True) #Turn the grid on
	plt.ylabel(ylabel) #Y-axis label
	plt.xlabel("Method") #X-axis label
	plt.title(ylabel +" vs Method - " +title) #Plot title
	plt.xlim(0.5,3.5) #set x axis range
	plt.ylim(0,maxy) #Set yaxis range

	#Set the bar labels
	plt.gca().set_xticks(inds) #label locations
	plt.gca().set_xticklabels(labels) #label values

	#Make sure labels and titles are inside plot area
	plt.tight_layout()

	#Save the chart
	plt.savefig("../Figures/"+title+ylabel+"_bar_chart.pdf")

	#Displays the charts.
	#You must close the plot window for the code following each show()
	#to continue to run
	#plt.show()
	##clear graph fpr next set
	plt.clf()


def line_graph_alpha_error(title, name1,name2,name3, x_data,y_data,maxy=None,ylabel=None):

	#Create values and labels for line graphs
	values = y_data
	inds   = x_data
	labels =[name1,name2,name3]

	flatteny = reduce(list.__add__, (list(mi) for mi in y_data))

	#Plot a line graph
	plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
	plt.plot(inds,values[0],'or-', linewidth=3) #Plot the first series in red with circle marker
	plt.plot(inds,values[1],'sb-', linewidth=3) #Plot the first series in blue with square marker
	plt.plot(inds,values[2],'^g-', linewidth=3) #Plot the first series in gren with ^ marker

	#This plots the data
	plt.grid(True) #Turn the grid on
	plt.ylabel("Error") #Y-axis label
	plt.xlabel("alpha Values") #X-axis label
	plt.title("Error vs alpha Value - " +title) #Plot title
	plt.xlim(-1,max(x_data)*1.1) #set x axis range
	plt.ticklabel_format(style='sci', axis='x')
	
	plt.ylim(0,max(flatteny)*1.1) #Set yaxis range
	plt.legend(labels,loc="best")

	#Make sure labels and titles are inside plot area
	plt.tight_layout()

	#Save the chart
	plt.savefig("../Figures/"+title+"_line_plot.pdf")

	#Displays the plots.
	#You must close the plot window for the code following each show()
	#to continue to run
	##plt.show()
	##clear graph fpr next set
	plt.clf()





