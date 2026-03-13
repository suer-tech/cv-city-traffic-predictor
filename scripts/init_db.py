from app.config import settings
from app.storage.db import Base, SessionLocal, engine
from app.storage.models import Camera, Route
from app.storage.repository import Repository


def main() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        repo = Repository(session)
        repo.upsert_camera(
            Camera(
                camera_id=settings.default_camera_id,
                source_url=settings.default_camera_url,
                intersection_id=settings.default_intersection_id,
                direction=settings.default_direction,
                is_enabled=True,
                roi_config={},
            )
        )
        repo.upsert_route(Route(route_id="route_default", name="Default route", is_enabled=True))
        repo.replace_route_cameras("route_default", [settings.default_camera_id])
        session.commit()


if __name__ == "__main__":
    main()
