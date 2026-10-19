
document.addEventListener("DOMContentLoaded", function () {

    /* =========================================================
       GET HTML ELEMENTS
    ========================================================= */

    const input =
        document.getElementById("mriInput");

    const previewContainer =
        document.getElementById(
            "imagePreviewContainer"
        );

    const preview =
        document.getElementById(
            "imagePreview"
        );

    const form =
        document.getElementById(
            "uploadForm"
        );

    const button =
        document.getElementById(
            "predictButton"
        );

    const loading =
        document.getElementById(
            "loadingMessage"
        );


    /* =========================================================
       CHECK FILE INPUT
    ========================================================= */

    if (!input) {

        console.error(
            "MRI file input #mriInput was not found."
        );

        return;
    }


    /* =========================================================
       FILE SELECTION
    ========================================================= */

    input.addEventListener(
        "change",
        function () {

            const file =
                this.files[0];


            /* No file selected */

            if (!file) {

                if (previewContainer) {

                    previewContainer
                        .classList
                        .add("d-none");

                }

                return;
            }


            /* =================================================
               CHECK FILE TYPE
            ================================================= */

            const allowedTypes = [
                "image/jpeg",
                "image/png"
            ];


            if (
                !allowedTypes.includes(
                    file.type
                )
            ) {

                alert(
                    "Please select a JPG, JPEG or PNG image."
                );

                this.value = "";


                if (previewContainer) {

                    previewContainer
                        .classList
                        .add("d-none");

                }

                return;
            }


            /* =================================================
               CHECK FILE SIZE
            ================================================= */

            const maxSize =
                10 * 1024 * 1024;


            if (file.size > maxSize) {

                alert(
                    "Image size must be less than 10 MB."
                );

                this.value = "";


                if (previewContainer) {

                    previewContainer
                        .classList
                        .add("d-none");

                }

                return;
            }


            /* =================================================
               READ IMAGE
            ================================================= */

            const reader =
                new FileReader();


            reader.onload =
                function (event) {

                    if (preview) {

                        preview.src =
                            event.target.result;

                    }


                    if (previewContainer) {

                        previewContainer
                            .classList
                            .remove("d-none");

                    }

                };


            reader.readAsDataURL(file);

        }
    );


    /* =========================================================
       FORM SUBMISSION
    ========================================================= */

    if (form) {

        form.addEventListener(
            "submit",
            function (event) {

                /* Check file */

                if (!input.files.length) {

                    event.preventDefault();

                    alert(
                        "Please select an MRI image first."
                    );

                    return;
                }


                /* Disable button */

                if (button) {

                    button.disabled = true;

                    button.innerHTML =
                        "Processing...";

                }


                /* Show loading message */

                if (loading) {

                    loading.classList
                        .remove("d-none");

                }

            }
        );

    }

});

