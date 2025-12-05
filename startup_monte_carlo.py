import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

st.set_page_config(page_title="VC Monte Carlo Simulator", layout="wide", page_icon="📊")

st.title("🚀 Startup Monte Carlo Simulator for VCs")
st.markdown("### Probabilistic modeling for investment decisions")

# Sidebar for inputs
st.sidebar.header("Investment Parameters")

# Investment basics
investment_amount = st.sidebar.number_input("Investment Amount ($M)", min_value=0.1, max_value=100.0, value=5.0, step=0.5)
ownership_pct = st.sidebar.slider("Ownership %", min_value=1.0, max_value=50.0, value=20.0, step=0.5)
current_valuation = st.sidebar.number_input("Current Valuation ($M)", min_value=1.0, max_value=500.0, value=25.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.header("Growth Assumptions")

# Revenue projections
current_arr = st.sidebar.number_input("Current ARR ($M)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)
growth_rate_mean = st.sidebar.slider("Mean Growth Rate (%/year)", min_value=0, max_value=300, value=100, step=10)
growth_rate_std = st.sidebar.slider("Growth Rate Std Dev (%)", min_value=0, max_value=100, value=30, step=5)

# Exit assumptions
st.sidebar.markdown("---")
st.sidebar.header("Exit Assumptions")
years_to_exit = st.sidebar.slider("Years to Exit", min_value=3, max_value=10, value=5, step=1)
exit_multiple_mean = st.sidebar.slider("Exit Multiple (Mean)", min_value=1.0, max_value=20.0, value=10.0, step=0.5)
exit_multiple_std = st.sidebar.slider("Exit Multiple (Std Dev)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

# Probability of failure
failure_rate_annual = st.sidebar.slider("Annual Failure Rate (%)", min_value=0, max_value=50, value=15, step=1) / 100

# Dilution
follow_on_rounds = st.sidebar.slider("Expected Follow-on Rounds", min_value=0, max_value=5, value=2, step=1)
dilution_per_round = st.sidebar.slider("Dilution per Round (%)", min_value=0, max_value=40, value=20, step=5) / 100

st.sidebar.markdown("---")
n_simulations = st.sidebar.number_input("Number of Simulations", min_value=1000, max_value=50000, value=10000, step=1000)

# Run simulation button
if st.sidebar.button("🎲 Run Simulation", type="primary"):
    with st.spinner("Running Monte Carlo simulation..."):
        np.random.seed(42)
        
        # Initialize arrays
        final_valuations = np.zeros(n_simulations)
        final_revenues = np.zeros(n_simulations)
        returns_multiple = np.zeros(n_simulations)
        cash_returns = np.zeros(n_simulations)
        irr_values = np.zeros(n_simulations)
        outcomes = []
        
        for i in range(n_simulations):
            # Simulate survival
            survived = True
            for year in range(years_to_exit):
                if np.random.random() < failure_rate_annual:
                    survived = False
                    break
            
            if not survived:
                final_valuations[i] = 0
                final_revenues[i] = 0
                returns_multiple[i] = 0
                cash_returns[i] = -investment_amount
                irr_values[i] = -1.0
                outcomes.append("Failed")
            else:
                # Simulate revenue growth
                revenue = current_arr
                for year in range(years_to_exit):
                    # Sample growth rate (using normal distribution, bounded at 0)
                    growth = max(0, np.random.normal(growth_rate_mean / 100, growth_rate_std / 100))
                    revenue *= (1 + growth)
                
                final_revenues[i] = revenue
                
                # Simulate exit multiple (lognormal distribution for better realism)
                mu = np.log(exit_multiple_mean) - 0.5 * (exit_multiple_std / exit_multiple_mean) ** 2
                sigma = exit_multiple_std / exit_multiple_mean
                exit_multiple = np.random.lognormal(mu, sigma)
                
                # Calculate exit valuation
                exit_valuation = revenue * exit_multiple
                final_valuations[i] = exit_valuation
                
                # Calculate dilution
                final_ownership = ownership_pct / 100
                for _ in range(follow_on_rounds):
                    final_ownership *= (1 - dilution_per_round)
                
                # Calculate returns
                exit_value = exit_valuation * final_ownership
                cash_returns[i] = exit_value - investment_amount
                returns_multiple[i] = exit_value / investment_amount
                
                # Calculate IRR
                if exit_value > investment_amount:
                    irr = (exit_value / investment_amount) ** (1 / years_to_exit) - 1
                    irr_values[i] = irr
                    outcomes.append("Success")
                else:
                    irr_values[i] = -((investment_amount - exit_value) / investment_amount) ** (1 / years_to_exit)
                    outcomes.append("Loss")
        
        # Store results in session state
        st.session_state['results'] = {
            'final_valuations': final_valuations,
            'final_revenues': final_revenues,
            'returns_multiple': returns_multiple,
            'cash_returns': cash_returns,
            'irr_values': irr_values,
            'outcomes': outcomes
        }

# Display results if available
if 'results' in st.session_state:
    results = st.session_state['results']
    
    # Key metrics
    st.markdown("---")
    st.header("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        median_multiple = np.median(results['returns_multiple'])
        st.metric("Median MOIC", f"{median_multiple:.2f}x", 
                 delta=f"{(median_multiple - 1) * 100:.0f}% return")
    
    with col2:
        median_irr = np.median(results['irr_values']) * 100
        st.metric("Median IRR", f"{median_irr:.1f}%")
    
    with col3:
        success_rate = (np.array(results['outcomes']) == "Success").sum() / len(results['outcomes']) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col4:
        expected_value = np.mean(results['cash_returns'])
        st.metric("Expected Value", f"${expected_value:.2f}M")
    
    # Percentile analysis
    st.markdown("---")
    st.header("📈 Return Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Return Multiple Percentiles")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(results['returns_multiple'], percentiles)
        
        perc_df = pd.DataFrame({
            'Percentile': [f"P{p}" for p in percentiles],
            'Return Multiple': [f"{v:.2f}x" for v in percentile_values],
            'Cash Return': [f"${(v * investment_amount - investment_amount):.2f}M" for v in percentile_values]
        })
        st.dataframe(perc_df, use_container_width=True)
    
    with col2:
        st.subheader("Outcome Distribution")
        outcome_counts = pd.Series(results['outcomes']).value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=outcome_counts.index,
            values=outcome_counts.values,
            hole=0.3,
            marker=dict(colors=['#ff4b4b', '#ffa600', '#00cc88'])
        )])
        fig_pie.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed visualizations
    st.markdown("---")
    st.header("📉 Detailed Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Return Distribution", "Revenue Projection", "Probability Analysis", "Sensitivity"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Return multiple distribution
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Histogram(
                x=results['returns_multiple'],
                nbinsx=50,
                name='Return Multiple',
                marker_color='#00cc88'
            ))
            fig_returns.add_vline(x=np.median(results['returns_multiple']), 
                                 line_dash="dash", line_color="red",
                                 annotation_text="Median")
            fig_returns.update_layout(
                title="Return Multiple Distribution (MOIC)",
                xaxis_title="Multiple on Invested Capital",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_returns, use_container_width=True)
        
        with col2:
            # IRR distribution
            fig_irr = go.Figure()
            fig_irr.add_trace(go.Histogram(
                x=results['irr_values'] * 100,
                nbinsx=50,
                name='IRR',
                marker_color='#0066ff'
            ))
            fig_irr.add_vline(x=np.median(results['irr_values']) * 100,
                             line_dash="dash", line_color="red",
                             annotation_text="Median")
            fig_irr.update_layout(
                title="IRR Distribution",
                xaxis_title="IRR (%)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_irr, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Exit valuation distribution
            fig_val = go.Figure()
            fig_val.add_trace(go.Histogram(
                x=results['final_valuations'],
                nbinsx=50,
                name='Exit Valuation',
                marker_color='#ffa600'
            ))
            fig_val.add_vline(x=np.median(results['final_valuations']),
                             line_dash="dash", line_color="red",
                             annotation_text="Median")
            fig_val.update_layout(
                title="Exit Valuation Distribution",
                xaxis_title="Exit Valuation ($M)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_val, use_container_width=True)
        
        with col2:
            # Revenue distribution
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Histogram(
                x=results['final_revenues'],
                nbinsx=50,
                name='Final Revenue',
                marker_color='#9c27b0'
            ))
            fig_rev.add_vline(x=np.median(results['final_revenues']),
                             line_dash="dash", line_color="red",
                             annotation_text="Median")
            fig_rev.update_layout(
                title="Exit ARR Distribution",
                xaxis_title="ARR at Exit ($M)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_rev, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Probability of achieving target returns
            st.subheader("Probability of Achieving Target Returns")
            targets = [1, 3, 5, 10, 20, 50, 100]
            probabilities = [
                (results['returns_multiple'] >= target).sum() / len(results['returns_multiple']) * 100
                for target in targets
            ]
            
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Bar(
                x=[f"{t}x" for t in targets],
                y=probabilities,
                marker_color='#00cc88',
                text=[f"{p:.1f}%" for p in probabilities],
                textposition='outside'
            ))
            fig_prob.update_layout(
                title="Probability of Achieving Return Targets",
                xaxis_title="Return Multiple Target",
                yaxis_title="Probability (%)",
                height=400
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        
        with col2:
            # Risk-return scatter
            st.subheader("Risk-Return Profile")
            
            # Create bins for visualization
            sample_size = min(1000, len(results['returns_multiple']))
            sample_indices = np.random.choice(len(results['returns_multiple']), sample_size, replace=False)
            
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=results['final_revenues'][sample_indices],
                y=results['returns_multiple'][sample_indices],
                mode='markers',
                marker=dict(
                    size=5,
                    color=results['returns_multiple'][sample_indices],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Return Multiple"),
                    opacity=0.6
                ),
                name='Simulations'
            ))
            fig_scatter.update_layout(
                title="Exit Revenue vs Return Multiple",
                xaxis_title="Exit ARR ($M)",
                yaxis_title="Return Multiple (x)",
                height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab4:
        st.subheader("Quick Sensitivity Analysis")
        st.markdown("Compare current assumptions to alternative scenarios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Scenario**")
            st.write(f"Growth Rate: {growth_rate_mean}% ± {growth_rate_std}%")
            st.write(f"Exit Multiple: {exit_multiple_mean}x ± {exit_multiple_std}x")
            st.write(f"Annual Failure Rate: {failure_rate_annual * 100}%")
        
        with col2:
            scenario = st.selectbox(
                "Compare with:",
                ["Bull Case", "Bear Case", "Base Case Adjusted"]
            )
            
            if scenario == "Bull Case":
                st.write(f"Growth Rate: {growth_rate_mean + 30}% ± {growth_rate_std - 5}%")
                st.write(f"Exit Multiple: {exit_multiple_mean + 2}x ± {exit_multiple_std}x")
                st.write(f"Annual Failure Rate: {(failure_rate_annual - 0.05) * 100}%")
            elif scenario == "Bear Case":
                st.write(f"Growth Rate: {growth_rate_mean - 30}% ± {growth_rate_std + 10}%")
                st.write(f"Exit Multiple: {exit_multiple_mean - 2}x ± {exit_multiple_std + 1}x")
                st.write(f"Annual Failure Rate: {(failure_rate_annual + 0.10) * 100}%")
    
    # Download results
    st.markdown("---")
    st.header("💾 Export Results")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Metric': [
            'Investment Amount ($M)',
            'Initial Ownership (%)',
            'Median Return Multiple',
            'Mean Return Multiple',
            'Median IRR (%)',
            'Mean IRR (%)',
            'Success Rate (%)',
            'Failure Rate (%)',
            'Expected Value ($M)',
            'P10 Return',
            'P90 Return',
            'Median Exit Valuation ($M)',
            'Median Exit ARR ($M)'
        ],
        'Value': [
            f"{investment_amount:.2f}",
            f"{ownership_pct:.1f}",
            f"{np.median(results['returns_multiple']):.2f}x",
            f"{np.mean(results['returns_multiple']):.2f}x",
            f"{np.median(results['irr_values']) * 100:.1f}",
            f"{np.mean(results['irr_values']) * 100:.1f}",
            f"{success_rate:.1f}",
            f"{((np.array(results['outcomes']) == 'Failed').sum() / len(results['outcomes']) * 100):.1f}",
            f"{expected_value:.2f}",
            f"{np.percentile(results['returns_multiple'], 10):.2f}x",
            f"{np.percentile(results['returns_multiple'], 90):.2f}x",
            f"{np.median(results['final_valuations']):.2f}",
            f"{np.median(results['final_revenues']):.2f}"
        ]
    })
    
    st.download_button(
        label="📥 Download Summary (CSV)",
        data=summary_df.to_csv(index=False),
        file_name="monte_carlo_summary.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Adjust parameters in the sidebar and click 'Run Simulation' to begin")
    
    # Show example use cases
    st.markdown("---")
    st.header("💡 Use Cases for VCs")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Portfolio Analysis**
        - Model expected returns across portfolio
        - Understand risk-adjusted returns
        - Calculate fund-level metrics
        """)
    
    with col2:
        st.markdown("""
        **Due Diligence**
        - Stress test growth assumptions
        - Evaluate downside scenarios
        - Compare deal terms
        """)
    
    with col3:
        st.markdown("""
        **LP Reporting**
        - Generate probabilistic forecasts
        - Show risk distributions
        - Support investment memos
        """)

st.markdown("---")
st.markdown("*Built for venture capital firms to make data-driven investment decisions*")
