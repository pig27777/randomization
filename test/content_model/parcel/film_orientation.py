"""Rotate a membrane, its contact probe and gravity together on all six faces."""
import argparse
import json
from pathlib import Path

import numpy as np
from film_checks import manifest
from six_face_contact import coverage


def main(output):
    output.mkdir(parents=True,exist_ok=False)
    axis=np.array([.3,.8,.5]);axis/=np.linalg.norm(axis)
    angle=.71
    cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    rotation=np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)
    gravity=np.array([.3,-.4,-9.81])
    manifest(output,rotation=rotation.tolist(),gravity=gravity.tolist(),fixture='native instantaneous contact acceleration')
    baseline=coverage(gravity=gravity)
    rotated=coverage(rotation=rotation,gravity=gravity)
    errors=[]
    for original,transformed in zip(baseline,rotated):
        expected=rotation@np.array(original['contact_force_world'])
        error=np.linalg.norm(np.array(transformed['contact_force_world'])-expected)
        errors.append(float(error))
        assert original['contacts']==transformed['contacts']>0
    result=dict(baseline=baseline,rotated=rotated,force_error_N=errors,max_error_N=max(errors))
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(max_error_N=max(errors),faces=len(errors))))
    assert max(errors)<1.e-6,result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
