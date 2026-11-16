"""GPU spring-only energy ledger with physical contact/stop losses excluded."""
import argparse
from unittest.mock import patch
import xml.etree.ElementTree as ET
from common import ROOT,setup,write
import numpy as np
import warp as wp
import mujoco_warp as mjw
from contents.parcel import Batch
from contents.parcel import kernels as K
from contents.parcel.model import xml as original_xml


def fixture_xml(*args):
    tree=ET.fromstring(original_xml(*args))
    tree.find('option').set('gravity','0 0 0')
    for geom in tree.iter('geom'):geom.set('contype','0');geom.set('conaffinity','0')
    for joint in tree.iter('joint'):
        if joint.get('limited')=='true':joint.set('limited','false')
    return ET.tostring(tree,encoding='unicode')


def main(output):
    out=setup(output,'GPU closed spring energy ledger')
    rows=[]
    for dt in (.00025,.0005,.001):
        for damping in (0.,.1):
            with patch('contents.parcel.model.xml',side_effect=fixture_xml):b=Batch(1,prototype=3,level=0,dt=dt)
            b.model.dof_damping.fill_(damping)
            q=b.data.qpos.numpy()
            q[0,b.cpu.joint('rot1_1').qposadr[0]]=.1
            q[0,b.cpu.joint('rot2_1').qposadr[0]]=-.05
            cap_q=b.ids.numpy()[:,9];q[0,cap_q[0]]=-.005;b.data.qpos.assign(q)
            mjw.forward(b.model,b.data)
            trace=[];diss=0.;p=b.params.numpy()[0]
            def measure(time):
                mjw.step1(b.model,b.data);mjw.energy_pos(b.model,b.data);mjw.energy_vel(b.model,b.data)
                native=b.data.energy.numpy()[0];positions=b.data.qpos.numpy()[0,cap_q]
                linear=.5*float(p['compression'])*np.sum(positions**2)
                quadratic=float(p['quadratic'])*np.sum(np.maximum(-positions,0)**3)/3
                return [time,float(native[1]),0.,float(native[0])-linear,linear,quadratic,diss,0.]
            trace.append(measure(0.));initial=sum(trace[0][1:6])
            for i in range(round(.1/dt)):
                # measure() has just refreshed current kinematics/passive forces.
                b.data.xfrc_applied.zero_();b.data.qfrc_applied.zero_()
                wp.launch(K.dynamics,1,[dt,0,0,b.cart,0,0,0,0,0.,b.params,b.states,b.ids,b.normal,b.gap,
                    b.data.xpos,b.data.xmat,b.data.xipos,b.data.qpos,b.data.qfrc_applied,b.data.xfrc_applied])
                mjw.step2(b.model,b.data)
                v=b.data.qvel.numpy()[0];diss+=float(damping*np.sum(v*v))*dt
                trace.append(measure((i+1)*dt))
            data=np.array(trace);balance=data[:,1:6].sum(1)+data[:,6]-initial
            rows.append(dict(dt=dt,damping=damping,initial_energy=initial,final_energy=float(data[-1,1:6].sum()),
                relative_residual=float(balance[-1]/initial),max_absolute_relative_residual=float(abs(balance).max()/initial),
                max_positive_relative_residual=float(max(balance.max(),0)/initial)))
            np.savez_compressed(out/f'gpu-energy-{dt}-c{damping}.npz',trace=data,balance=balance,params=b.params.numpy(),
                columns=['time','KE','GPE','joint_U','compression_linear_U','compression_quadratic_U','dissipation','external_work'])
    write(out/'summary.json',rows)
    print(out/'summary.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
