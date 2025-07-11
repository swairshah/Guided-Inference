import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and prepare the data
def load_data(file_path):
    df = pd.read_json(file_path, lines=True)
    df = df.explode('analysis_result')
    df = pd.concat([df.drop(['analysis_result'], axis=1), df['analysis_result'].apply(pd.Series)], axis=1)
    return df

# Function to plot correlation heatmap
def plot_correlation_heatmap(df):
    corr = df[['acc', 'entropy', 'varentropy', 'prob_top2_diff']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()

# Function to plot parameter distributions by accuracy
def plot_parameter_distributions(df, parameter):
    plt.figure(figsize=(10, 6))
    # Create a copy of the dataframe with reset index to avoid duplicate labels
    plot_df = df[['acc', parameter]].reset_index(drop=True)
    sns.histplot(data=plot_df, x=parameter, hue='acc', kde=True, element='step', stat='density', common_norm=False)
    plt.title(f'Distribution of {parameter} by Accuracy')
    plt.savefig(f'{parameter}_distribution.png')
    plt.close()

# Function to plot entropy scatter plot
def plot_entropy_scatter(df):
    plt.figure(figsize=(12, 8))
    colors = ['red' if acc == 0 else 'blue' for acc in df['acc']]
    plt.scatter(df['position'], df['entropy'], c=colors, alpha=0.5)
    plt.xlabel('Position')
    plt.ylabel('Entropy')
    plt.title('Entropy by Position (Red: Accuracy 0, Blue: Accuracy 1)')
    plt.savefig('entropy_scatter.png')
    plt.close()

# Main function
def main():
    # Load the data
    df = load_data('analysis_c10_k50.csv')  

    plot_correlation_heatmap(df)

    # Plot parameter distributions
    for param in ['entropy', 'varentropy', 'prob_top2_diff']:
        plot_parameter_distributions(df, param)

    # Plot entropy scatter plot
    plot_entropy_scatter(df)

    # Calculate and print average values for each parameter by accuracy
    print(df.groupby('acc')[['entropy', 'varentropy', 'prob_top2_diff']].mean())

if __name__ == "__main__":
    main()
