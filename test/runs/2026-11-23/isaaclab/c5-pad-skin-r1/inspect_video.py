"""Compose matched clocks and extract verification frames from the encoded deliverables."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import subprocess as sp

from PIL import Image, ImageDraw, ImageFont


def ffmpeg(*args):
    return sp.check_output(["ffmpeg", "-loglevel", "error", "-n", *map(str,args)])


def frame(path, number):
    data = ffmpeg("-i",path,"-vf",f"select=eq(n\\,{number})","-frames:v",1,
                  "-f","image2pipe","-vcodec","png","-")
    return Image.open(BytesIO(data)).convert("RGB")


def compare(before, after, out):
    image = Image.new("RGB",(1440,906),"#1d2a32")
    draw = ImageDraw.Draw(image)
    for row, number in enumerate((84,116)):
        for col, (path,label) in enumerate(((before,"Original envelope"),(after,"Measured pad / visual skin"))):
            snap = frame(path,number).crop((427,168,973,489)).resize((720,423),Image.Resampling.LANCZOS)
            origin = (col*720,row*453)
            image.paste(snap,(origin[0],origin[1]+30))
            draw.text((origin[0]+12,origin[1]+3),f"{label}   |   frame {number}   {number/25:.2f} s",
                      font=ImageFont.truetype("DejaVuSans.ttf",19),fill="white")
    image.save(out)


def compose(left,right,output):
    filters = []
    for channel,(title,subtitle) in enumerate((
        ("MuJoCo flex","Two-sheet PE shell  /  compressible three-segment insert"),
        ("Low-order C5","PPO training parcel configuration  /  scripted top press"))):
        filters.append(f"[{channel}:v]crop=856:642:376:44,scale=800:600,pad=800:676:0:76:color=0x1d2a32,"
                       f"drawtext=font=DejaVuSans:fontsize=26:fontcolor=white:x=16:y=9:text='{title}',"
                       f"drawtext=font=DejaVuSans:fontsize=16:fontcolor=0xd4e1df:x=16:y=45:text='{subtitle}'[p{channel}]")
    filters.append("[p0][p1]hstack=inputs=2,drawbox=x=799:y=0:w=2:h=ih:c=0x465861:t=fill")
    ffmpeg("-i",left,"-i",right,"-filter_complex",";".join(filters),"-c:v","libx264","-crf",17,
           "-pix_fmt","yuv420p","-movflags","+faststart",output)
    fields = "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames,duration"
    probe = sp.check_output(["ffprobe","-v","error","-count_frames","-show_entries",fields,
                             "-of","json",str(output)])
    stream = json.loads(probe)["streams"][0]
    if (stream["nb_read_frames"],stream["r_frame_rate"],stream["width"],stream["height"]) != ("180","25/1",1600,676):
        raise RuntimeError(f"Unexpected final stream: {stream}")
    output.with_suffix(".ffprobe.json").write_bytes(probe)
    ffmpeg("-xerror","-i",output,"-f","null","-")
    sheet=Image.new("RGB",(1120,3*498),"#1d2a32")
    for row,(i,phase) in enumerate(((45,"Approach"),(84,"Press"),(145,"Retraction"))):
        shot=frame(output,i).resize((1120,473),Image.Resampling.LANCZOS)
        sheet.paste(shot,(0,row*498+25))
        ImageDraw.Draw(sheet).text((10,row*498+2),f"{phase}   |   {i/25:.2f} s   |   frame {i}",
            font=ImageFont.truetype("DejaVuSans.ttf",17),fill="white")
    sheet.save(output.with_suffix(".keyframes.png"))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("original",type=Path)
    p.add_argument("simple",type=Path)
    p.add_argument("flex",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    compare(a.original,a.simple,a.output.with_suffix(".before-after.png"))
    compose(a.flex,a.simple,a.output)
