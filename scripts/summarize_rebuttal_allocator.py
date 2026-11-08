"""Read allocator metadata only; replay live allocations and retain stack evidence."""
import collections
import gzip
import json
from pathlib import Path


def blocks(snapshot):
    result = {}
    for segment in snapshot.get('segments', []):
        address = segment['address']
        for block in segment['blocks']:
            pointer = block.get('address', address)
            if block['state'] == 'active_allocated':
                result[pointer] = (block['size'], block.get('frames', []))
            address += block['size']
    return result


def evidence(live):
    grouped = collections.Counter()
    for size, frames in live.values():
        stack = ' | '.join(f"{f.get('filename', '')}:{f.get('line', '')}:{f.get('name', '')}"
                           for f in frames if f.get('filename', '').endswith('.py'))
        grouped[stack or 'unattributed (no Python stack)'] += size
    return [dict(bytes=size, stack=stack) for stack, size in grouped.most_common(15)]


def summarize(path):
    with gzip.open(path, 'rt') as handle:
        data = json.load(handle)
    initial = blocks(data['initial_snapshot'])
    final = blocks(data['snapshot'])
    before = blocks(data['before_optimizer_snapshot'])
    live = dict(initial)
    current = sum(v[0] for v in live.values())
    peak, peak_live = current, dict(live)
    unknown_frees = 0
    traces = data['snapshot'].get('device_traces', [])
    actions = collections.Counter()
    for trace in traces:
        for event in trace:
            action = event['action']; actions[action] += 1
            pointer = event.get('addr')
            if action == 'alloc':
                old = live.pop(pointer, (0, []))
                current -= old[0]
                live[pointer] = (event['size'], event.get('frames', []))
                current += event['size']
            elif action == 'free_requested':
                old = live.pop(pointer, None)
                if old is None:
                    unknown_frees += 1
                else:
                    current -= old[0]
            if current > peak:
                peak, peak_live = current, dict(live)
    final_bytes = sum(v[0] for v in final.values())
    return dict(path=str(path), step=data['step'], initial_allocated_bytes=sum(v[0] for v in initial.values()),
                before_optimizer_allocated_bytes=sum(v[0] for v in before.values()),
                final_allocated_bytes=final_bytes, replay_final_bytes=current,
                replay_peak_bytes=peak, replay_matches_final=current == final_bytes,
                unknown_frees=unknown_frees, actions=dict(actions),
                peak_stack_evidence=evidence(peak_live),
                inventory={k:v['bytes'] for k,v in data['inventory'].items()},
                attribution_note='Stack-group bytes at one replayed peak; semantic labels require interpretation. Not a sum of phase maxima.')


def inspect(root):
    for path in sorted(Path(root).glob('**/allocator-rank*-step*.json.gz')):
        if 'interrupted' not in path.parts:
            print('ALLOCATOR_SUMMARY=' + json.dumps(summarize(path)), flush=True)
