"""Opt-in memory attribution. Does not modify model or optimizer arithmetic."""
import collections
import atexit
import bisect
import gzip
import json
import os
from pathlib import Path
import threading
import subprocess
import time


def block_map(snapshot):
    live = {}
    for segment in snapshot['segments']:
        address = segment['address']
        for block in segment['blocks']:
            pointer = block.get('address', address)
            if block['state'] == 'active_allocated':
                live[pointer] = dict(size=block['size'], frames=block.get('frames', []),
                                     born=-1, requested=block.get('requested_size'))
            address += block['size']
    return live


def replay(initial, final, inventories, before_index, measured_peak):
    """Reconstruct live bytes and reserved segments at one allocated-memory peak."""
    start_index = len(initial.get('device_traces', [[]])[0])
    events = final['device_traces'][0][start_index:]
    live = block_map(initial)
    pending = {b.get('address',0):dict(size=b['size']) for segment in initial['segments']
               for b in segment['blocks'] if b['state']=='active_awaiting_free'}
    occupied = {**live, **pending}
    addresses = sorted(occupied)
    segments = {s['address']: s['total_size'] for s in initial['segments']}
    labels = {}
    # Attach inventories to allocation lifetimes, not raw pointers across reuse.
    def label_at(index):
        for category, records in inventories.get(index, {}).items():
            for address, size in records.items():
                address = int(address)
                if address in live:
                    labels.setdefault((address, live[address]['born']), (category, size))
    label_at(-1)
    current = sum(x['size'] for x in live.values())
    peak = current
    peak_live, peak_segments = dict(live), dict(segments)
    peak_pending = dict(pending)
    peak_index = -1
    unknown = 0
    reserved_supported = True
    for index, event in enumerate(events):
        action, pointer = event['action'], event.get('addr')
        if action == 'alloc':
            if pointer in live:
                raise RuntimeError('Allocation address reused while live')
            # Native ALLOC traces contain requested bytes, not allocated block size.
            # Reconstruct the default native allocator's rounding and split rule.
            segment_end = next(base+size for base,size in segments.items() if base <= pointer < base+size)
            position = bisect.bisect_left(addresses,pointer)
            limit = min(segment_end,addresses[position]) if position < len(addresses) else segment_end
            available = limit-pointer
            rounded = max(512, (event['size']+511)//512*512)
            if available < rounded:
                raise RuntimeError('Allocator replay has insufficient free block')
            split = available-rounded >= 512 if rounded <= 1048576 else available-rounded > 1048576
            size = rounded if split else available
            live[pointer] = dict(size=size, frames=event.get('frames', []),born=index,requested=event['size'])
            occupied[pointer]=live[pointer]
            bisect.insort(addresses,pointer)
            current += size
        elif action == 'free_requested':
            old = live.pop(pointer, None)
            if old is None:
                unknown += 1
            else:
                current -= old['size']
                pending[pointer]=old
        elif action == 'free_completed':
            pending.pop(pointer,None)
            if pointer in occupied:
                occupied.pop(pointer)
                addresses.pop(bisect.bisect_left(addresses,pointer))
        elif action == 'segment_alloc':
            segments[pointer] = event['size']
        elif action == 'segment_free':
            segments.pop(pointer, None)
        elif action in ('segment_map', 'segment_unmap'):
            reserved_supported = False
        label_at(index)
        if current > peak:
            peak, peak_live, peak_segments, peak_index = current, dict(live), dict(segments), index
            peak_pending = dict(pending)
    # Inventories correspond to snapshot event counts (boundary after last event).
    totals = collections.Counter()
    stacks = collections.Counter()
    for pointer, block in peak_live.items():
        identity = (pointer, block['born'])
        if identity in labels:
            category, storage_bytes = labels[identity]
            occupied = min(storage_bytes, block['size'])
            totals[category] += occupied
            totals['persistent_allocation_padding'] += block['size'] - occupied
        else:
            category = ('optimizer_phase_unclassified' if block['born'] >= before_index
                        else 'forward_backward_or_preexisting_unclassified')
            totals[category] += block['size']
            frames = [f for f in block['frames'] if f.get('filename', '').endswith('.py')]
            stack = ' | '.join(f"{Path(f['filename']).name}:{f.get('line')}:{f.get('name')}" for f in frames[:7])
            stacks[stack or 'no_python_stack'] += block['size']
    split_free, cached_free = 0,0
    for base,size in peak_segments.items():
        used=sum(b['size'] for address,b in {**peak_live,**peak_pending}.items() if base <= address < base+size)
        if used:split_free += size-used
        else:cached_free += size
    final_bytes = sum(v['size'] for v in block_map(final).values())
    return dict(peak_allocated_bytes=peak, peak_event_index=peak_index,
                requested_at_peak_bytes=sum(b['requested'] for b in peak_live.values()),
                allocation_padding_at_peak_bytes=sum(b['size']-b['requested'] for b in peak_live.values()),
                inactive_split_at_peak_bytes=split_free, inactive_unsplit_at_peak_bytes=cached_free,
                pending_free_at_peak_bytes=sum(b['size'] for b in peak_pending.values()),
                peak_phase='optimizer' if peak_index >= before_index else 'forward_backward',
                reserved_at_peak_bytes=sum(peak_segments.values()) if reserved_supported else None,
                components_at_peak=dict(totals),
                unclassified_stack_bytes=[dict(stack=k,bytes=v) for k,v in stacks.most_common(12)],
                replay_final_bytes=current, snapshot_final_bytes=final_bytes,
                replay_matches_final=current == final_bytes, unknown_frees=unknown,
                matches_measured_peak=peak == measured_peak, measured_peak_bytes=measured_peak,
                trace_events=len(events), reserved_replay_supported=reserved_supported,
                note='Components share one peak instant. Unclassified bytes are not automatically optimizer workspace. Default native allocator block splitting is replayed, including pending frees. Cross-check final and peak bytes before using inferred padding/split values.')


def tensor_storages(values):
    import torch
    seen, result = set(), {}
    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor) and value.is_cuda:
            storage = value.untyped_storage()
            result[storage.data_ptr()] = storage.nbytes()
        elif isinstance(value, dict):
            for child in value.values(): visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value: visit(child)
        else:
            for attribute in ('basis','ortho_matrix','transformed_low_rank'):
                if hasattr(value,attribute):visit(getattr(value,attribute))
    visit(values)
    return result


class Tracker:
    def __init__(self, inventory):
        import torch
        self.torch = torch
        self.inventory = inventory
        self.root = Path(os.environ['ATTRIBUTION_DIR'])
        self.root.mkdir(parents=True, exist_ok=True)
        runtime=dict(pid=os.getpid(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
                     gpu_name=torch.cuda.get_device_name(),allocator_config=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))
        try:
            runtime['compute_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
        except Exception as error:
            runtime['process_inventory_error']=repr(error)
        (self.root/'runtime.json').write_text(json.dumps(runtime,indent=2))
        self.traced_steps = set(range(1,16)) if os.environ.get('ATTRIBUTION_TRACE') == '1' else set()
        self.raw_peak = -1
        self.step = 0
        self.phase = 'setup'
        self.lock = threading.Lock()
        self.sampled = None
        self.stop = threading.Event()
        self.sampler = threading.Thread(target=self.sample_loop, daemon=True)
        self.sampler.start()
        atexit.register(self.close)

    def close(self):
        self.stop.set()
        if threading.current_thread() is not self.sampler:
            self.sampler.join()

    def counters(self):
        cuda = self.torch.cuda
        stats = cuda.memory_stats()
        free, total = cuda.mem_get_info()
        return dict(timestamp=time.time(), allocated=cuda.memory_allocated(), reserved=cuda.memory_reserved(),
                    device_used=total-free, inactive_split=stats.get('inactive_split_bytes.all.current'),
                    active=stats.get('active_bytes.all.current'), requested=stats.get('requested_bytes.all.current'),
                    allocator_backend=cuda.memory.get_allocator_backend())

    def sample_loop(self):
        self.torch.cuda.set_device(0)
        while not self.stop.wait(.01):
            try:
                value = self.counters()
                value.update(step=self.step, phase=self.phase)
                with self.lock:
                    if self.sampled is None or value['allocated'] > self.sampled['allocated']:
                        self.sampled = value
            except Exception as error:
                self.sampling_error = repr(error)
                return

    def begin(self):
        self.step += 1
        self.phase = 'forward_backward'
        with self.lock: self.sampled = None
        self.trace = self.step in self.traced_steps
        if self.trace:
            self.torch.cuda.memory._record_memory_history(max_entries=1000000)
            self.initial = self.torch.cuda.memory._snapshot()
            self.start_index = len(self.initial['device_traces'][0])
            self.inventories = {-1: self.inventory()}
        self.start = self.counters()

    def before_optimizer(self):
        self.torch.cuda.synchronize()
        self.before = self.counters()
        self.fb_peak = self.torch.cuda.max_memory_allocated()
        self.phase = 'optimizer'
        if self.trace:
            snapshot = self.torch.cuda.memory._snapshot()
            self.before_index = len(snapshot['device_traces'][0]) - self.start_index
            self.inventories[self.before_index - 1] = self.inventory()

    def finish(self, peak, loss):
        self.torch.cuda.synchronize()
        phase_peak = peak
        peak = max(peak, self.torch.cuda.max_memory_allocated())
        end = self.counters()
        inventory = self.inventory()
        result = dict(step=self.step, losses=loss, peak_allocated_bytes=peak,
                      pre_finalize_phase_peak_bytes=phase_peak,
                      peak_scope="full step including loss reduction/finalization",
                      forward_backward_peak_bytes=self.fb_peak,
                      optimizer_peak_bytes=self.torch.cuda.max_memory_allocated(),
                      start=self.start, before_optimizer=self.before, end=end,
                      persistent_bytes={k:sum(v.values()) for k,v in inventory.items()},
                      sampled_near_peak=self.sampled, sampling_error=getattr(self,'sampling_error',None),
                      sample_note='10ms sampling; device-used and allocator counters are sequential approximate observations, not guaranteed exact global peak.')
        if self.trace:
            final = self.torch.cuda.memory._snapshot()
            self.torch.cuda.memory._record_memory_history(enabled=None)
            self.inventories[len(final['device_traces'][0])-self.start_index-1] = inventory
            result['attribution'] = replay(self.initial, final, self.inventories, self.before_index, peak)
            stat=os.statvfs(self.root)
            if stat.f_bavail*stat.f_frsize <= 7_000_000_000:
                raise RuntimeError('CRITICAL CLOUD.RU STORAGE: insufficient reserve plus 2GB snapshot write allowance')
            paths=[]
            if self.step in (1,6,11):paths.append(self.root/f'snapshot-step{self.step}.json.gz')
            if peak > self.raw_peak:
                self.raw_peak=peak
                paths.append(self.root/'snapshot-peak.json.gz')
            for path in paths:
                with gzip.open(path,'wt') as handle:
                    json.dump(dict(initial=self.initial,final=final,inventories=self.inventories,before_index=self.before_index,measured_peak=peak,step=self.step),handle)
        with (self.root/'telemetry.jsonl').open('a') as handle:
            handle.write(json.dumps(result)+'\n')
        self.phase = 'between_steps'
        if self.step == 15:self.close()
