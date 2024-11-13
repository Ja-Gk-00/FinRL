import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from tqdm import tqdm
class UtilsVis:

    @staticmethod
    def visualize_agent_logs(filename, agent_name):
        df = pd.read_csv(filename)
        
        plt.figure(figsize=(14, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(df['Step'], df['Portfolio Value'], label='Portfolio Value')
        plt.title(f'{agent_name} Portfolio Value Over Time')
        plt.ylabel('Portfolio Value')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(df['Step'], df['Action'], label='Action', color='green')
        plt.yticks([0, 1, 2], ['Buy', 'Sell', 'Hold'])
        plt.ylabel('Action')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(df['Step'], df['Reward'], label='Reward', color='orange')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_portfolio_gif(log_filename, gif_filename, agent_name, sample_rate=10):
        df = pd.read_csv(log_filename)
        if sample_rate > 1:
            df = df.iloc[::sample_rate].reset_index(drop=True)
        
        steps = df['Step']
        portfolio = df['Portfolio Value']

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f'{agent_name} Portfolio Value Over Time')
        ax.set_xlabel('Step')
        ax.set_ylabel('Portfolio Value ($)')
        line, = ax.plot([], [], lw=2, color='blue')
        ax.grid(True)
        text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        ax.set_xlim(steps.min(), steps.max())
        buffer = (portfolio.max() - portfolio.min()) * 0.05
        ax.set_ylim(portfolio.min() - buffer, portfolio.max() + buffer)

        def init():
            line.set_data([], [])
            text.set_text('')
            return line, text

        def update(frame):
            current_step = steps[frame]
            current_value = portfolio[frame]
            line.set_data(steps[:frame+1], portfolio[:frame+1])
            text.set_text(f'Step: {current_step}\nPortfolio: ${current_value:.2f}')
            return line, text
        
        total_frames = len(df)

        ani = FuncAnimation(
            fig, update, frames=range(total_frames),
            init_func=init, blit=True, repeat=False
        )

        try:
            ani.save(gif_filename, writer='pillow', fps=30)
            print(f"GIF successfully saved as {gif_filename}")
        except Exception as e:
            print(f"An error occurred while saving the GIF: {e}")
        finally:
            plt.close(fig)


