import pandas as pd
import matplotlib.pyplot as plt

# Set Matplotlib backend to an interactive one
plt.switch_backend('TkAgg')

def analyze_csv(file_path):
    data = pd.read_csv(file_path)

    # Group data by AP/Router ID
    router_groups = data.groupby('AP/Router ID')

    results = {}

    for router_id, group in router_groups:
        total_packets = len(group)
        lost_packets = len(group[group['Status'] == 'Lost'])
        received_packets = group[group['Status'] == 'Received']
        percentage_lost_packets = (lost_packets / total_packets) * 100 if total_packets > 0 else 0
        average_delay = received_packets['Delay'].mean() if not received_packets.empty else 0
        average_throughput = received_packets['Throughput (Mbps)'].mean() if not received_packets.empty else 0

        results[router_id] = {
            'total_packets': total_packets,
            'lost_packets': lost_packets,
            'percentage_lost_packets': percentage_lost_packets,
            'average_delay': average_delay,
            'average_throughput': average_throughput
        }

    return results

def plot_results(results1, results2, csv_file1, csv_file2):
    routers = sorted(results1.keys())

    # Prepare data for plotting
    total_packets1 = [results1[router]['total_packets'] for router in routers]
    total_packets2 = [results2[router]['total_packets'] for router in routers]
    lost_packets1 = [results1[router]['lost_packets'] for router in routers]
    lost_packets2 = [results2[router]['lost_packets'] for router in routers]
    avg_delay1 = [results1[router]['average_delay'] for router in routers]
    avg_delay2 = [results2[router]['average_delay'] for router in routers]
    avg_throughput1 = [results1[router]['average_throughput'] for router in routers]
    avg_throughput2 = [results2[router]['average_throughput'] for router in routers]
    percentage_lost_packets1 = [results1[router]['percentage_lost_packets'] for router in routers]
    percentage_lost_packets2 = [results2[router]['percentage_lost_packets'] for router in routers]

    x = range(len(routers))

    # Plotting total packets
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    width = 0.35  # width of the bars

    axs[0].bar(x, total_packets1, width, label=csv_file1)
    axs[0].bar([p + width for p in x], total_packets2, width, label=csv_file2)
    axs[0].set_ylabel('Total Packets')
    axs[0].set_title('Total Packets per Router')
    axs[0].set_xticks([p + width / 2 for p in x])
    axs[0].set_xticklabels(routers)
    axs[0].legend()

    # Plotting lost packets percentage
    axs[1].bar(x, percentage_lost_packets1, width, label=csv_file1)
    axs[1].bar([p + width for p in x], percentage_lost_packets2, width, label=csv_file2)
    axs[1].set_ylabel('Percentage Lost Packets')
    axs[1].set_title('Percentage Lost Packets per Router')
    axs[1].set_xticks([p + width / 2 for p in x])
    axs[1].set_xticklabels(routers)
    axs[1].legend()

    # Plotting average delay and throughput
    width = 0.2
    axs[2].bar(x, avg_delay1, width, label=f'Avg Delay - {csv_file1}')
    axs[2].bar([p + width for p in x], avg_delay2, width, label=f'Avg Delay - {csv_file2}')
    axs[2].bar([p + 2 * width for p in x], avg_throughput1, width, label=f'Avg Throughput - {csv_file1}')
    axs[2].bar([p + 3 * width for p in x], avg_throughput2, width, label=f'Avg Throughput - {csv_file2}')
    axs[2].set_ylabel('Time (s) / Throughput (Mbps)')
    axs[2].set_title('Average Delay and Throughput per Router')
    axs[2].set_xticks([p + 1.5 * width for p in x])
    axs[2].set_xticklabels(routers)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('comparison_plots.png')
    plt.show()

def compare_simulations(csv_file1, csv_file2):
    results1 = analyze_csv(csv_file1)
    results2 = analyze_csv(csv_file2)

    for router_id in results1:
        print(f"Results for Router {router_id}:")
        data = {
            'Metric': ['Total Packets', 'Lost Packets', 'Percentage Lost Packets', 'Average Delay', 'Average Throughput'],
            csv_file1: [
                results1[router_id]['total_packets'],
                results1[router_id]['lost_packets'],
                f"{results1[router_id]['percentage_lost_packets']:.2f}%",
                f"{results1[router_id]['average_delay']:.6f} seconds",
                f"{results1[router_id]['average_throughput']:.6f} Mbps"
            ],
            csv_file2: [
                results2[router_id]['total_packets'],
                results2[router_id]['lost_packets'],
                f"{results2[router_id]['percentage_lost_packets']:.2f}%",
                f"{results2[router_id]['average_delay']:.6f} seconds",
                f"{results2[router_id]['average_throughput']:.6f} Mbps"
            ]
        }

        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        print()

    plot_results(results1, results2, csv_file1, csv_file2)

if __name__ == "__main__":
    csv_file1 = "combined_router_data.csv"
    csv_file2 = "combined_router_data_cwmin.csv"

    compare_simulations(csv_file1, csv_file2)
