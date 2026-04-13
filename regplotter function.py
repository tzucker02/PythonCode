def regplotter(df, feature1, feature1_title, feature2, feature2_title, feature3, feature3_title):
    # Accept either a string column name or a one-item list like ['col_name'].
    f1 = feature1[0] if isinstance(feature1, (list, tuple)) else feature1
    f2 = feature2[0] if isinstance(feature2, (list, tuple)) else feature2
    f3 = feature3[0] if isinstance(feature3, (list, tuple)) else feature3

    featurelist = [f1, f2, f3]
    df_clean = df.dropna(subset=featurelist)

    # Set style
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 8)

    plt.figure(figsize=(10, 8))

    # Create scatter plot
    scatter = sns.scatterplot(
        data=df_clean, 
        x=f1, 
        y=f2,
        hue=feature3,
        palette='viridis', 
        alpha=0.7, 
        s=60,
        edgecolor='k',
        legend=False
    )

    # Add regression line (using all data points, not colored by state)
    reg_line = sns.regplot(
        data=df_clean, 
        x=f1, 
        y=f2, 
        scatter=False,  # Don't show the scatter points again
        color='red', 
        line_kws={'linewidth': 2.5, 'label': 'Regression Line'},
        ci=95,  # Show 95% confidence interval
    )

    # Calculate and display regression statistics.
    x_values = df_clean[f1].to_numpy(dtype=float)
    y_values = df_clean[f2].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_values, y_values, 1)
    r_value = np.corrcoef(x_values, y_values)[0, 1]
    r_squared = float(r_value ** 2)
    p_value = float("nan")

    # Add text annotation with regression statistics
    text_str = f'Regression Statistics:\nSlope: {slope:.2f}\nR²: {r_squared:.3f}\nP-value: {p_value:.4f}'
    plt.text(0.80, 0.15, text_str, transform=plt.gca().transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.title(f'Relationship Between {feature1_title} and {feature2_title} (with Regression Analysis)', fontsize=16)
    plt.xlabel(f'{feature1_title} ({f1})', fontsize=12)
    plt.ylabel(f'{feature2_title} ({f2})', fontsize=12)
    plt.axhline(0, color='darkgray', linestyle='--', linewidth=1.5, label='Break-even Point')

    plt.tight_layout()
    plt.show()

    # Optional: Print detailed regression output
    print("=" * 60)
    print("REGRESSION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Dependent Variable:  {feature2_title} ({f2})")
    print(f"Independent Variable: {feature1_title} ({f1})")
    print(f"\nRegression Equation: y = {intercept:.2f} + ({slope:.2f})x")
    print(f"R-squared: {r_squared:.3f}")
    print(f"P-value: {p_value:.4f}")
    print(f"\nInterpretation:")
    print(f"- For every 1-unit increase in {f1}, {f2} changes by {slope:.2f}")
    print(f"- R² of {r_squared:.3f} indicates {'strong' if r_squared > 0.5 else 'moderate' if r_squared > 0.2 else 'weak'} correlation")
    print(f"- P-value {'< 0.05 (statistically significant)' if p_value < 0.05 else '> 0.05 (not statistically significant)'}")
    print("=" * 60)
    
    return slope, intercept, r_squared, p_value