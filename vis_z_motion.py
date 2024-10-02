import csv
import plotly.graph_objs as go
import plotly.io as pio

# Path to CSV file
csv_path = "motion.csv"

# Initialize lists for data
x = []
y = []
y2 = []

# Read CSV data
with open(csv_path, 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    first_y, first_y2 = None, None
    for row in lines:
        break
    for row in lines:
        if not first_y and not first_y2:
            first_y = float(row[2])
            first_y2 = float(row[7])
        x.append(float(row[8]))
        y.append(float(row[2])-first_y)
        y2.append(float(row[7])-first_y2)



# Create traces for the two plots
trace1 = go.Scatter(x=x, y=y, mode='lines', name='Line 1')
trace2 = go.Scatter(x=x, y=y2, mode='lines', name='Line 2')

# Create a figure with the two traces
fig = go.Figure(data=[trace1, trace2])

# Show the plot
pio.show(fig)
