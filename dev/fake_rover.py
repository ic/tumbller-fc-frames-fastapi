from datetime import datetime
import io

from fastapi import FastAPI, Response
from PIL import Image, ImageDraw, ImageFont
import uvicorn


def make_image(rover: str, ts: datetime, width: int = 300, colour: tuple = (255, 255, 255)) -> Image:
    img = Image.new("RGB", (width, int(width * 9 / 16)))
    lines = [f"Rover {rover}", ts.strftime("%Y-%m-%d %H:%M:%S")]
    font = ImageFont.load_default(30)
    canvas = ImageDraw.Draw(img)
    bbwidth = max([canvas.textlength(t, font) for t in lines])
    bbheight = len(lines) * 45
    anchor = ((img.width - bbwidth) // 2, (img.height - bbheight) // 2)
    canvas.multiline_text(
        anchor, "\n".join(lines), font=font, fill=colour, align="center"
    )
    return img


app = FastAPI()


@app.get("/cameras/{rover}")
async def cameras(rover: str):
    buffer = io.BytesIO()
    img = make_image(rover, datetime.now())
    img.save(buffer, format="JPEG")
    img.close()
    return Response(buffer.getvalue(), media_type="image/jpeg")


@app.get("/tumbllers/{rover}/motor/{command}")
async def tumbllers_motor(rover: str, command: str):
    return {}


if __name__ == "__main__":
    uvicorn.run(
        "dev.fake_rover:app",
        host="0.0.0.0",
        port=5001,
        reload=True,
    )
